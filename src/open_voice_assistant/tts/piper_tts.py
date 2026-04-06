"""Streaming text-to-speech using piper."""

import asyncio
import logging
from pathlib import Path
from collections.abc import AsyncIterator

import numpy as np

from open_voice_assistant.config import Settings
from open_voice_assistant.tts import TTS, resample

logger = logging.getLogger(__name__)

MODELS_DIR = Path.home() / ".cache" / "piper-voices"


class PiperTTS(TTS):
    """Text-to-speech engine using piper with sentence-level streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voice = None
        self.sample_rate: int = 16000  # output rate (resampled from model native)
        self.sample_width: int = 2  # 16-bit
        self.channels: int = 1  # mono

    def load(self) -> None:
        from piper import PiperVoice

        model_path = self._resolve_model(self._settings.tts_model)
        logger.info("Loading piper TTS model: %s", model_path)
        self._voice = PiperVoice.load(str(model_path), use_cuda=False)
        logger.info("TTS model loaded (native %dHz, output %dHz)",
                     self._voice.config.sample_rate, self.sample_rate)

    def _resolve_model(self, model_name: str) -> Path:
        if Path(model_name).exists():
            return Path(model_name)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.onnx"

        if not model_path.exists():
            logger.info("Downloading piper voice: %s", model_name)
            from piper.download_voices import download_voice
            download_voice(model_name, MODELS_DIR)
            if not model_path.exists() or model_path.stat().st_size == 0:
                raise RuntimeError(f"Download failed for {model_name}")

        return model_path

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate.

        CPU-bound — callers should run in an executor.
        """
        if self._voice is None:
            raise RuntimeError("Call load() first")

        all_audio = []
        for chunk in self._voice.synthesize(text):
            audio = chunk.audio_int16_array
            if chunk.sample_rate != self.sample_rate:
                audio = resample(audio, chunk.sample_rate, self.sample_rate)
            all_audio.append(audio)

        if not all_audio:
            return b""

        return np.concatenate(all_audio).tobytes()

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Stream TTS: accumulate text into sentences, synthesize each eagerly."""
        sentence_buf = ""
        full_response = ""
        sentence_endings = ".!?"
        loop = asyncio.get_running_loop()

        async for chunk in text_chunks:
            sentence_buf += chunk
            if len(full_response) < 200:
                full_response += chunk

            while True:
                split_idx = -1
                for i, ch in enumerate(sentence_buf):
                    if ch in sentence_endings and i > 10:
                        if i + 1 >= len(sentence_buf) or sentence_buf[i + 1] == " ":
                            split_idx = i + 1
                            break

                if split_idx == -1:
                    break

                sentence = sentence_buf[:split_idx].strip()
                sentence_buf = sentence_buf[split_idx:].lstrip()

                sentence = sentence.replace("[LISTEN]", "").strip()
                if sentence:
                    logger.debug("TTS synthesizing: %r", sentence[:60])
                    audio = await loop.run_in_executor(None, self.synthesize, sentence)
                    yield audio

        remainder = sentence_buf.strip()
        # Strip control tokens that shouldn't be spoken
        remainder = remainder.replace("[LISTEN]", "").strip()
        if remainder:
            logger.debug("TTS synthesizing remainder: %r", remainder[:60])
            audio = await loop.run_in_executor(None, self.synthesize, remainder)
            yield audio

        logger.debug("Agent response: %r", full_response[:200])
