"""Streaming text-to-speech using Qwen3-TTS-Streaming-ONNX (voice cloning)."""

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

MODELS_DIR = CACHE_DIR / "qwen3-tts"

_HF_BASE = "https://huggingface.co/pltobing/Qwen3-TTS-Streaming-ONNX/resolve/main"

_FILES: list[tuple[str, str]] = [
    (
        "qwen3-tts_onnx/talker_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/talker_model.onnx",
    ),
    (
        "qwen3-tts_onnx/talker_local_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/talker_local_model.onnx",
    ),
    (
        "qwen3-tts_onnx/codec_decoder_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/codec_decoder_model.onnx",
    ),
    (
        "qwen3-tts_onnx/speaker_encoder_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/speaker_encoder_model.onnx",
    ),
    (
        "qwen3-tts_onnx/talker_codec_embed_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/talker_codec_embed_model.onnx",
    ),
    (
        "qwen3-tts_onnx/text_embed_proj_model.onnx",
        f"{_HF_BASE}/qwen3-tts_onnx/text_embed_proj_model.onnx",
    ),
    ("configs/vocab.json", f"{_HF_BASE}/configs/vocab.json"),
    ("configs/merges.txt", f"{_HF_BASE}/configs/merges.txt"),
    ("configs/tokenizer_config.json", f"{_HF_BASE}/configs/tokenizer_config.json"),
    ("configs/config.json", f"{_HF_BASE}/configs/config.json"),
    (
        "configs/preprocessor_config.json",
        f"{_HF_BASE}/configs/preprocessor_config.json",
    ),
    (
        "configs/speech_tokenizer_config.json",
        f"{_HF_BASE}/configs/speech_tokenizer_config.json",
    ),
    (
        "audio_ref/female_shadowheart.flac",
        f"{_HF_BASE}/audio_ref/female_shadowheart.flac",
    ),
]

_NATIVE_RATE = 24000
_DEFAULT_REF_REL = "audio_ref/female_shadowheart.flac"
_CHUNK_FRAMES = 4  # 0.32s; bg-threaded codec keeps this off the critical path
_FP32_MODEL_DIR = "qwen3-tts_onnx"
_INT8_MODEL_DIR = "qwen3-tts_onnx_int8"  # produced by python -m ovi_voice_assistant.tts.qwen3.quantize


class Qwen3TTS(TTS):
    """Streaming TTS via in-house Qwen3 inferencer with bg-threaded codec."""

    def __init__(self, settings: Settings, sample_rate: int = 16000) -> None:
        self._settings = settings
        self._inferencer = None
        self.sample_rate: int = sample_rate
        self.sample_width: int = 2
        self.channels: int = 1

    def load(self) -> None:
        from ovi_voice_assistant.tts.qwen3 import Qwen3Inferencer

        self._ensure_files()

        ref_setting = self._settings.tts.reference_audio
        ref_path = (
            Path(ref_setting).expanduser()
            if ref_setting
            else MODELS_DIR / _DEFAULT_REF_REL
        )

        providers = self._select_providers()

        int8_dir = MODELS_DIR / _INT8_MODEL_DIR
        fp32_dir = MODELS_DIR / _FP32_MODEL_DIR
        model_dir = int8_dir if int8_dir.exists() else fp32_dir

        logger.info(
            "Loading Qwen3-TTS (ref=%s, lang=%s, weights=%s, providers=%s)",
            ref_path.name,
            self._settings.tts.language,
            "int8" if model_dir == int8_dir else "fp32",
            [p if isinstance(p, str) else p[0] for p in providers],
        )

        self._inferencer = Qwen3Inferencer(
            model_dir=model_dir,
            configs_dir=MODELS_DIR / "configs",
            ref_audio_path=ref_path,
            language=self._settings.tts.language,
            providers=providers,
            num_threads=os.cpu_count() or 4,
            chunk_frames=_CHUNK_FRAMES,
        )

        logger.info(
            "Qwen3-TTS ready (native %dHz, output %dHz)",
            _NATIVE_RATE,
            self.sample_rate,
        )

    @staticmethod
    def _select_providers() -> list:
        """CPU-only by default; CUDA when available.

        CoreML fails to compile the talker LLM (dynamic KV-cache shapes hit
        MLProgram error -7), so it is intentionally not selected here.
        """
        available = set(ort.get_available_providers())
        providers: list = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _ensure_files() -> None:
        for rel, url in _FILES:
            dest = MODELS_DIR / rel
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s ...", rel)
            urllib.request.urlretrieve(url, dest)
            logger.info("Downloaded %s (%d MB)", rel, dest.stat().st_size // 1_000_000)

    def _emit(self, wav: np.ndarray) -> bytes:
        if wav.size == 0:
            return b""
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
        if self.sample_rate != _NATIVE_RATE:
            pcm = resample(pcm, _NATIVE_RATE, self.sample_rate)
        return pcm.tobytes()

    def synthesize_iter(self, text: str) -> Iterable[bytes]:
        """Yield 16-bit PCM chunks (~0.32s each) at self.sample_rate."""
        if self._inferencer is None:
            raise RuntimeError("Call load() first")

        for wav in self._inferencer.synthesize(text):
            data = self._emit(wav)
            if data:
                yield data

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate."""
        return b"".join(self.synthesize_iter(text))
