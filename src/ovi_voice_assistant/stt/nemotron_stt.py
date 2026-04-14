"""Streaming speech-to-text using Nemotron Speech ONNX (FastConformer RNNT).

Uses the ONNX export of nvidia/nemotron-speech-streaming-en-0.6b with int8
quantization for CPU inference. True streaming: processes 560ms audio chunks,
emitting text tokens as speech arrives.

Model: https://huggingface.co/danielbodart/nemotron-speech-600m-onnx
"""

import asyncio
import json
import logging
import queue as queue_mod
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.stt.stt import STT, VadStartCallback

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "danielbodart/nemotron-speech-600m-onnx"

# RNNT blank token (vocab_size = 1024, blank = 1024)
BLANK_ID = 1024
MAX_SYMBOLS_PER_FRAME = 10

# Streaming parameters
POLL_TIMEOUT_S = 0.2
NO_SPEECH_TIMEOUT_S = 5.0
MAX_LISTEN_S = 60.0

# End-of-speech: consecutive encoder chunks with no tokens after speech.
# Each chunk is 560ms, so 2 chunks ≈ 1.1s of silence.
SILENCE_CHUNKS_THRESHOLD = 2


class NemotronSTT(STT):
    """Streaming speech-to-text using Nemotron ONNX FastConformer RNNT.

    Processes 560ms audio chunks through a cache-aware streaming encoder and
    RNNT greedy decoder. No external VAD needed — end-of-speech is detected
    via consecutive blank-only encoder chunks.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name = settings.stt_model
        if "/" not in self._model_name:
            self._model_name = DEFAULT_MODEL

        self._encoder: ort.InferenceSession | None = None
        self._decoder: ort.InferenceSession | None = None
        self._vocab: list[str] = []
        self._filterbank: np.ndarray | None = None  # [n_mels, n_fft//2+1]
        self._window: np.ndarray | None = None

        # Config (populated from model's config.json)
        self._n_mels = 128
        self._n_fft = 512
        self._hop_length = 160
        self._win_length = 400
        self._preemph = 0.97
        self._chunk_mel_frames = 56
        self._pre_encode_cache_frames = 9
        self._chunk_samples = 8960  # 560ms at 16kHz

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        variant = (
            "int8-dynamic" if self._settings.stt_device == "cpu" else "int8-static"
        )

        logger.info("Loading Nemotron STT: %s (%s)", self._model_name, variant)

        model_dir = Path(
            snapshot_download(
                self._model_name,
                allow_patterns=["config.json", "shared/*", f"{variant}/*"],
            )
        )

        # Load and apply config
        with open(model_dir / "config.json") as f:
            config = json.load(f)

        prep = config["preprocessor"]
        self._n_mels = prep["n_mels"]
        self._n_fft = prep["n_fft"]
        self._hop_length = prep["hop_length"]
        self._win_length = prep["win_length"]
        self._preemph = prep["preemph"]

        enc = config["encoder"]
        self._chunk_mel_frames = enc["chunk_mel_frames"]
        self._pre_encode_cache_frames = enc["pre_encode_cache_frames"]
        self._chunk_samples = config["streaming"]["chunk_audio_samples"]

        # Load filterbank [1, n_mels, n_fft//2+1] → [n_mels, n_fft//2+1]
        fb_path = model_dir / "shared" / "filterbank.bin"
        n_bins = self._n_fft // 2 + 1
        self._filterbank = np.frombuffer(
            fb_path.read_bytes(), dtype=np.float32
        ).reshape(1, self._n_mels, n_bins)[0]

        # Load vocabulary (format: "token_text id" per line)
        tokens_path = model_dir / "shared" / "tokens.txt"
        self._vocab = []
        for line in tokens_path.read_text().strip().split("\n"):
            token = line.rsplit(" ", 1)[0]
            self._vocab.append(token)

        # Hann window
        self._window = np.hanning(self._win_length).astype(np.float32)

        # ONNX sessions
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._settings.stt_device != "cpu"
            else ["CPUExecutionProvider"]
        )
        self._encoder = ort.InferenceSession(
            str(model_dir / variant / "encoder_model.onnx"),
            providers=providers,
        )
        self._decoder = ort.InferenceSession(
            str(model_dir / variant / "decoder_model.onnx"),
            providers=providers,
        )

        logger.info(
            "Nemotron STT loaded (variant=%s, vocab=%d, chunk=%dms)",
            variant,
            len(self._vocab),
            self._chunk_samples * 1000 // 16000,
        )

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    def _apply_preemph(
        self, audio: np.ndarray, prev_sample: float = 0.0
    ) -> tuple[np.ndarray, float]:
        """Apply pre-emphasis filter. Returns (filtered, last_raw_sample)."""
        last = float(audio[-1]) if len(audio) > 0 else prev_sample
        out = np.empty_like(audio)
        out[0] = audio[0] - self._preemph * prev_sample
        out[1:] = audio[1:] - self._preemph * audio[:-1]
        return out, last

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram from pre-emphasized audio.

        Args:
            audio: float32 array of samples (pre-emphasized).
        Returns:
            [n_mels, n_frames] float32 log-mel spectrogram.
        """
        n_frames = max(0, (len(audio) - self._win_length) // self._hop_length + 1)
        if n_frames == 0:
            return np.zeros((self._n_mels, 0), dtype=np.float32)

        # Frame extraction via stride tricks
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, self._win_length),
            strides=(audio.strides[0] * self._hop_length, audio.strides[0]),
        ).copy()

        frames *= self._window
        fft = np.fft.rfft(frames, n=self._n_fft)  # [n_frames, n_fft//2+1]
        power = np.abs(fft) ** 2
        mel = self._filterbank @ power.T  # [n_mels, n_frames]
        np.maximum(mel, 2**-24, out=mel)
        return np.log(mel).astype(np.float32)

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        pieces = [self._vocab[t] for t in token_ids if 0 <= t < len(self._vocab)]
        return "".join(pieces).replace("\u2581", " ").strip()

    # ------------------------------------------------------------------
    # RNNT decoder
    # ------------------------------------------------------------------

    def _run_rnnt_decoder(
        self, encoded: np.ndarray, encoded_len: int, decoder_state: dict
    ) -> list[int]:
        """Greedy RNNT decode over encoder output frames.

        Args:
            encoded: [1, dim, n_frames] encoder output.
            encoded_len: number of valid frames.
            decoder_state: mutable dict with states_1, states_2, last_token.
        Returns:
            List of emitted (non-blank) token IDs.
        """
        tokens: list[int] = []

        for i in range(encoded_len):
            frame = encoded[:, :, i : i + 1]  # [1, 1024, 1]
            symbols = 0

            while symbols < MAX_SYMBOLS_PER_FRAME:
                out = self._decoder.run(
                    None,
                    {
                        "encoder_outputs": frame,
                        "targets": np.array(
                            [[decoder_state["last_token"]]], dtype=np.int32
                        ),
                        "target_length": np.array([1], dtype=np.int32),
                        "input_states_1": decoder_state["states_1"],
                        "input_states_2": decoder_state["states_2"],
                    },
                )

                logits = out[0]  # [1, 1, 1, vocab_size+1]
                decoder_state["states_1"] = out[2]
                decoder_state["states_2"] = out[3]

                token_id = int(np.argmax(logits[0, 0, 0]))
                if token_id == BLANK_ID:
                    break

                tokens.append(token_id)
                decoder_state["last_token"] = token_id
                symbols += 1

        return tokens

    # ------------------------------------------------------------------
    # Encoder step (shared by transcribe and streaming)
    # ------------------------------------------------------------------

    def _encoder_step(
        self,
        chunk_mel: np.ndarray,
        mel_cache: np.ndarray,
        enc_cache: dict,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Run one encoder step.

        Returns:
            (encoded, encoded_len, new_mel_cache)
        """
        # Prepend mel cache → [1, n_mels, total_frames]
        mel_input = np.concatenate([mel_cache, chunk_mel], axis=1)[np.newaxis]
        length = np.array([mel_input.shape[2]], dtype=np.int64)

        out = self._encoder.run(
            None,
            {
                "audio_signal": mel_input,
                "length": length,
                "cache_last_channel": enc_cache["channel"],
                "cache_last_time": enc_cache["time"],
                "cache_last_channel_len": enc_cache["channel_len"],
            },
        )

        encoded = out[0]  # [1, dim, n_encoded]
        encoded_len = int(out[1][0])
        enc_cache["channel"] = out[2]
        enc_cache["time"] = out[3]
        enc_cache["channel_len"] = out[4]

        new_mel_cache = chunk_mel[:, -self._pre_encode_cache_frames :]
        return encoded, encoded_len, new_mel_cache

    def _init_encoder_cache(self) -> dict:
        return {
            "channel": np.zeros((1, 24, 70, 1024), dtype=np.float32),
            "time": np.zeros((1, 24, 1024, 8), dtype=np.float32),
            "channel_len": np.array([0], dtype=np.int64),
        }

    def _init_decoder_state(self) -> dict:
        return {
            "states_1": np.zeros((2, 1, 640), dtype=np.float32),
            "states_2": np.zeros((2, 1, 640), dtype=np.float32),
            "last_token": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe a complete audio buffer (CPU-bound)."""
        if self._encoder is None:
            raise RuntimeError("Call load() first")

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < self._chunk_samples * 0.1:
            return ""

        audio, _ = self._apply_preemph(audio)
        mel = self._compute_mel(audio)  # [n_mels, n_frames]

        mel_cache = np.zeros(
            (self._n_mels, self._pre_encode_cache_frames), dtype=np.float32
        )
        enc_cache = self._init_encoder_cache()
        dec_state = self._init_decoder_state()
        all_tokens: list[int] = []

        for start in range(0, mel.shape[1], self._chunk_mel_frames):
            chunk_mel = mel[:, start : start + self._chunk_mel_frames]
            if chunk_mel.shape[1] < self._chunk_mel_frames:
                chunk_mel = np.pad(
                    chunk_mel,
                    ((0, 0), (0, self._chunk_mel_frames - chunk_mel.shape[1])),
                )

            encoded, enc_len, mel_cache = self._encoder_step(
                chunk_mel, mel_cache, enc_cache
            )
            all_tokens.extend(self._run_rnnt_decoder(encoded, enc_len, dec_state))

        return self._decode_tokens(all_tokens)

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        on_vad_start: VadStartCallback | None = None,
    ) -> str:
        """Stream audio, returning text on end-of-speech."""
        if self._encoder is None:
            raise RuntimeError("Call load() first")

        audio_queue: queue_mod.Queue[bytes | None] = queue_mod.Queue()
        loop = asyncio.get_running_loop()

        async def _producer():
            async for chunk in audio_chunks:
                audio_queue.put(chunk)
            audio_queue.put(None)

        def _consumer() -> tuple[str, bool]:
            return self._run_streaming(audio_queue, loop, on_vad_start)

        producer_task = asyncio.create_task(_producer())
        try:
            text, speech_detected = await loop.run_in_executor(None, _consumer)
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

        if not speech_detected:
            return ""

        logger.debug("Nemotron STT: %r", text)
        return text

    def _run_streaming(
        self,
        audio_queue: queue_mod.Queue[bytes | None],
        loop: asyncio.AbstractEventLoop,
        on_vad_start: VadStartCallback | None,
    ) -> tuple[str, bool]:
        """Process audio queue through encoder + RNNT decoder. Runs in executor."""
        overlap = self._win_length - self._hop_length  # 240 samples

        # Audio buffer: starts with `overlap` zeros for first STFT window
        audio_buf = np.zeros(overlap, dtype=np.float32)
        preemph_prev = 0.0
        pcm_buf = b""

        mel_cache = np.zeros(
            (self._n_mels, self._pre_encode_cache_frames), dtype=np.float32
        )
        enc_cache = self._init_encoder_cache()
        dec_state = self._init_decoder_state()

        all_tokens: list[int] = []
        speech_detected = False
        vad_start_fired = False
        silent_chunks = 0
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if not speech_detected and elapsed > NO_SPEECH_TIMEOUT_S:
                logger.info("No speech after %.0fs, giving up", elapsed)
                break
            if elapsed > MAX_LISTEN_S:
                logger.warning("Max listen duration reached (%.0fs)", elapsed)
                break

            try:
                chunk = audio_queue.get(timeout=POLL_TIMEOUT_S)
            except queue_mod.Empty:
                continue

            if chunk is not None:
                pcm_buf += chunk

            # Convert complete samples and apply pre-emphasis
            n_bytes = (len(pcm_buf) // 2) * 2
            if n_bytes >= 2:
                raw = (
                    np.frombuffer(pcm_buf[:n_bytes], dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                pcm_buf = pcm_buf[n_bytes:]
                preemphed, preemph_prev = self._apply_preemph(raw, preemph_prev)
                audio_buf = np.concatenate([audio_buf, preemphed])

            # Process complete 560ms chunks
            end_of_speech = False
            while len(audio_buf) >= overlap + self._chunk_samples and not end_of_speech:
                mel_audio = audio_buf[: overlap + self._chunk_samples]
                audio_buf = audio_buf[self._chunk_samples :]  # keep overlap tail

                chunk_mel = self._compute_mel(mel_audio)

                # Pad/trim to exact chunk_mel_frames
                if chunk_mel.shape[1] < self._chunk_mel_frames:
                    chunk_mel = np.pad(
                        chunk_mel,
                        ((0, 0), (0, self._chunk_mel_frames - chunk_mel.shape[1])),
                    )
                elif chunk_mel.shape[1] > self._chunk_mel_frames:
                    chunk_mel = chunk_mel[:, : self._chunk_mel_frames]

                encoded, enc_len, mel_cache = self._encoder_step(
                    chunk_mel, mel_cache, enc_cache
                )
                chunk_tokens = self._run_rnnt_decoder(encoded, enc_len, dec_state)

                if chunk_tokens:
                    if not vad_start_fired and on_vad_start is not None:
                        asyncio.run_coroutine_threadsafe(on_vad_start(), loop)
                        vad_start_fired = True
                    speech_detected = True
                    silent_chunks = 0
                    all_tokens.extend(chunk_tokens)
                elif speech_detected:
                    silent_chunks += 1
                    if silent_chunks >= SILENCE_CHUNKS_THRESHOLD:
                        logger.info("End of speech (%d silent chunks)", silent_chunks)
                        end_of_speech = True

            if end_of_speech or chunk is None:
                break

        return self._decode_tokens(all_tokens), speech_detected
