"""Streaming speech-to-text using faster-whisper with Silero VAD."""

import asyncio
import io
import logging
from collections.abc import AsyncIterator

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model

from open_voice_assistant.config import Settings
from open_voice_assistant.stt import STT

logger = logging.getLogger(__name__)

# Silero VAD parameters
VAD_THRESHOLD = 0.4  # speech probability threshold
SILENCE_TIMEOUT_S = 1.0  # silence after speech to stop
MIN_SPEECH_S = 0.3  # minimum speech to be valid
MAX_LISTEN_S = 60.0  # maximum listen duration before forcing transcription
NO_SPEECH_TIMEOUT_S = 5.0  # give up if no speech detected within this time
VAD_CHUNK_SAMPLES = 512  # Silero expects 512-sample chunks at 16kHz


class WhisperSTT(STT):
    """Speech-to-text engine using faster-whisper with Silero VAD."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._vad_model = None

    def load(self) -> None:
        logger.info("Loading faster-whisper model: %s", self._settings.stt_model)
        self._model = WhisperModel(
            self._settings.stt_model,
            device="cpu",
            compute_type=self._settings.stt_compute_type,
        )
        self._vad_model = get_vad_model()
        logger.info("STT model loaded (with Silero VAD)")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio buffer. CPU-bound — callers should use run_in_executor."""
        if self._model is None:
            raise RuntimeError("Call load() first")

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < self._settings.mic_sample_rate * 0.1:
            return ""

        segments, info = self._model.transcribe(
            audio,
            language=self._settings.stt_language,
            beam_size=self._settings.stt_beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments)
        logger.debug("Transcribed: %r (lang=%s prob=%.2f)", text, info.language, info.language_probability)
        return text

    async def transcribe_stream(self, audio_chunks: AsyncIterator[bytes]) -> str:
        """Collect audio with Silero VAD to detect end-of-speech, then transcribe."""
        buf = io.BytesIO()
        vad_buf = b""  # accumulate bytes until we have a full VAD chunk
        speech_detected = False
        speech_bytes = 0
        silence_start: float | None = None
        loop = asyncio.get_running_loop()
        chunk_bytes = VAD_CHUNK_SAMPLES * 2  # 16-bit = 2 bytes per sample

        # Silero VAD state (hidden state for the RNN)
        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        context = np.zeros((1, 64), dtype="float32")

        listen_start = loop.time()

        async for chunk in audio_chunks:
            buf.write(chunk)
            vad_buf += chunk

            elapsed = loop.time() - listen_start
            if elapsed > MAX_LISTEN_S:
                logger.warning("Max listen duration reached (%.0fs)", MAX_LISTEN_S)
                break
            if not speech_detected and elapsed > NO_SPEECH_TIMEOUT_S:
                logger.info("No speech detected after %.0fs, giving up", NO_SPEECH_TIMEOUT_S)
                break

            # Process complete VAD chunks (512 samples = 1024 bytes)
            while len(vad_buf) >= chunk_bytes:
                vad_chunk = vad_buf[:chunk_bytes]
                vad_buf = vad_buf[chunk_bytes:]

                # Run Silero VAD on this chunk
                samples = np.frombuffer(vad_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                audio_with_context = np.concatenate([context[0], samples]).reshape(1, -1)
                context = samples.reshape(1, -1)[..., -64:]

                output, h, c = self._vad_model.session.run(
                    None, {"input": audio_with_context, "h": h, "c": c}
                )
                speech_prob = float(output[0])

                if speech_prob > VAD_THRESHOLD:
                    speech_detected = True
                    speech_bytes += chunk_bytes
                    silence_start = None
                elif speech_detected:
                    if silence_start is None:
                        silence_start = loop.time()
                    elif loop.time() - silence_start >= SILENCE_TIMEOUT_S:
                        speech_duration = speech_bytes / (self._settings.mic_sample_rate * self._settings.mic_sample_width)
                        if speech_duration >= MIN_SPEECH_S:
                            logger.info("End of speech (%.1fs speech, %.1fs silence)", speech_duration, loop.time() - silence_start)
                            break
                        else:
                            speech_detected = False
                            speech_bytes = 0
                            silence_start = None
                            buf = io.BytesIO()
            else:
                continue
            break  # inner while broke out — propagate to outer async for

        audio_bytes = buf.getvalue()
        if not audio_bytes:
            return ""

        return await loop.run_in_executor(None, self.transcribe, audio_bytes)
