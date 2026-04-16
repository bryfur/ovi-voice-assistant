"""Streaming text-to-speech using piper."""

import logging
from pathlib import Path

import numpy as np

from ovi_voice_assistant.config import CACHE_DIR, Settings
from ovi_voice_assistant.tts import resample
from ovi_voice_assistant.tts.tts import TTS

logger = logging.getLogger(__name__)

MODELS_DIR = CACHE_DIR / "piper"


class PiperTTS(TTS):
    """Text-to-speech engine using piper with sentence-level streaming."""

    def __init__(self, settings: Settings, sample_rate: int = 16000) -> None:
        self._settings = settings
        self._voice = None
        self._target_rate = sample_rate  # 0 = native model rate
        self.sample_rate: int = sample_rate
        self.sample_width: int = 2  # 16-bit
        self.channels: int = 1  # mono

    def load(self) -> None:
        from piper import PiperVoice

        model_path = self._resolve_model(self._settings.tts.model)
        logger.info("Loading piper TTS model: %s", model_path)
        self._voice = PiperVoice.load(str(model_path), use_cuda=False)
        native_rate = self._voice.config.sample_rate
        if self._target_rate == 0:
            self.sample_rate = native_rate
        logger.info(
            "TTS model loaded (native %dHz, output %dHz)", native_rate, self.sample_rate
        )

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
