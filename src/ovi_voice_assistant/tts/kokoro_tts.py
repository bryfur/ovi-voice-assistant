"""Streaming text-to-speech using Kokoro 82M (ONNX int8)."""

import logging
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ovi_voice_assistant.config import CACHE_DIR, Settings
from ovi_voice_assistant.tts import resample
from ovi_voice_assistant.tts.tts import TTS

logger = logging.getLogger(__name__)

MODELS_DIR = CACHE_DIR / "kokoro"

_MODEL_URL = (
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX"
    "/resolve/main/onnx/model_uint8.onnx"
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

        model_path = self._ensure_file("model_uint8.onnx", _MODEL_URL)
        voices_path = self._ensure_file("voices-v1.0.bin", _VOICES_URL)

        logger.info("Loading Kokoro TTS model")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        threads = os.cpu_count() or 4
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1

        providers = self._select_providers()
        session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
        self._kokoro = Kokoro.from_session(session, str(voices_path))

        self._input_dtypes = {
            inp.name: inp.type for inp in self._kokoro.sess.get_inputs()
        }

        logger.info(
            "Kokoro TTS loaded (providers=%s, native %dHz, output %dHz)",
            self._kokoro.sess.get_providers(),
            _NATIVE_RATE,
            self.sample_rate,
        )

        self._warmup()

    @staticmethod
    def _select_providers() -> list[str]:
        available = set(ort.get_available_providers())
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def _warmup(self) -> None:
        assert self._kokoro is not None
        voice = self._settings.tts.model or "af_heart"
        lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
        try:
            phonemes = self._kokoro.tokenizer.phonemize("Hello.", lang)
            for batch in self._kokoro._split_phonemes(phonemes):
                self._run_inference(batch, voice, 1.0)
        except Exception as e:
            logger.warning("Kokoro warmup failed: %s", e)

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

    def synthesize_iter(self, text: str) -> Iterable[bytes]:
        """Yield PCM for each phoneme batch as its inference completes.

        Kokoro splits long sentences into ~100-phoneme batches run as
        separate ORT forward passes. Yielding per batch lets the device
        start playing batch 1 while batch 2 is still inferring.
        """
        if self._kokoro is None:
            raise RuntimeError("Call load() first")

        voice = self._settings.tts.model or "af_heart"
        lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
        phonemes = self._kokoro.tokenizer.phonemize(text, lang)

        for batch in self._kokoro._split_phonemes(phonemes):
            samples = self._run_inference(batch, voice, 1.0)
            audio = (samples * 32767).clip(-32768, 32767).astype(np.int16)
            if self.sample_rate != _NATIVE_RATE:
                audio = resample(audio, _NATIVE_RATE, self.sample_rate)
            yield audio.tobytes()

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate."""
        return b"".join(self.synthesize_iter(text))
