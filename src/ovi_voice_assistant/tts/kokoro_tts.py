"""Streaming text-to-speech using Kokoro 82M (ONNX int8)."""

import asyncio
import logging
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ovi_voice_assistant.config import CACHE_DIR, Settings
from ovi_voice_assistant.tts import resample, split_sentences
from ovi_voice_assistant.tts.tts import TTS

logger = logging.getLogger(__name__)

MODELS_DIR = CACHE_DIR / "kokoro"

_MODEL_URL = (
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX"
    "/resolve/main/onnx/model_quantized.onnx"
)
_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx"
    "/releases/download/model-files-v1.0/voices-v1.0.bin"
)

_NATIVE_RATE = 24000


class KokoroTTS(TTS):
    """Text-to-speech using Kokoro 82M (ONNX int8 quantized).

    Uses kokoro-onnx for phonemization, tokenization, and voice loading.
    Runs ONNX inference directly to handle input type requirements of
    quantized models (kokoro-onnx passes wrong dtypes for some inputs).
    """

    def __init__(self, settings: Settings, sample_rate: int = 16000) -> None:
        self._settings = settings
        self._kokoro = None
        self._input_dtypes: dict[str, str] = {}
        self._target_rate = sample_rate
        self.sample_rate: int = sample_rate
        self.sample_width: int = 2  # 16-bit
        self.channels: int = 1  # mono

    def load(self) -> None:
        from kokoro_onnx import Kokoro

        model_path = self._ensure_file("model_quantized.onnx", _MODEL_URL)
        voices_path = self._ensure_file("voices-v1.0.bin", _VOICES_URL)

        logger.info("Loading Kokoro TTS model")
        self._kokoro = Kokoro(str(model_path), str(voices_path))

        # Cache expected input types so we can cast correctly
        self._input_dtypes = {
            inp.name: inp.type for inp in self._kokoro.sess.get_inputs()
        }

        logger.info(
            "Kokoro TTS loaded (native %dHz, output %dHz)",
            _NATIVE_RATE,
            self.sample_rate,
        )

    @staticmethod
    def _ensure_file(filename: str, url: str) -> Path:
        """Download a file to the cache directory if not already present."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / filename
        if not path.exists():
            logger.info("Downloading %s ...", filename)
            urllib.request.urlretrieve(url, path)
            logger.info(
                "Downloaded %s (%d MB)", filename, path.stat().st_size // 1_000_000
            )
        return path

    def _onnx_dtype(self, name: str) -> type[np.floating | np.integer]:
        """Return the numpy dtype the ONNX model expects for a given input."""
        onnx_type = self._input_dtypes.get(name, "")
        if "float" in onnx_type:
            return np.float32
        if "int64" in onnx_type:
            return np.int64
        return np.int32

    def _run_inference(
        self, phonemes: str, voice_name: str, speed: float
    ) -> np.ndarray:
        """Run ONNX inference with proper input types for quantized models."""
        tokens = self._kokoro.tokenizer.tokenize(phonemes)
        voice = self._kokoro.get_voice_style(voice_name)
        style = voice[len(tokens)]
        padded = [0, *tokens, 0]

        inputs = {
            "input_ids": np.array([padded], dtype=self._onnx_dtype("input_ids")),
            "style": np.array(style, dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
        }
        return self._kokoro.sess.run(None, inputs)[0]

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate."""
        if self._kokoro is None:
            raise RuntimeError("Call load() first")

        voice = self._settings.tts.model or "af_heart"
        lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
        phonemes = self._kokoro.tokenizer.phonemize(text, lang)

        audio_parts = []
        for batch in self._kokoro._split_phonemes(phonemes):
            part = self._run_inference(batch, voice, 1.0)
            audio_parts.append(part)

        samples = np.concatenate(audio_parts)
        audio = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        if self.sample_rate != _NATIVE_RATE:
            audio = resample(audio, _NATIVE_RATE, self.sample_rate)
        return audio.tobytes()

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream TTS: split tokens into sentences, synthesize each eagerly."""
        loop = asyncio.get_running_loop()
        async for sentence in split_sentences(text_chunks):
            logger.debug("TTS synthesizing: %r", sentence[:60])
            audio = await loop.run_in_executor(None, self.synthesize, sentence)
            yield audio
