"""Tests for ovi_voice_assistant.tts.qwen3_tts."""

from collections.abc import Iterable
from unittest.mock import patch

import numpy as np
import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.tts.qwen3_tts import Qwen3TTS


@pytest.fixture
def settings(tmp_path):
    with patch("ovi_voice_assistant.config.CONFIG_PATH", tmp_path / "c.yaml"):
        return Settings(_env_file=None, devices="")


class FakeInferencer:
    """Stub for Qwen3Inferencer matching only the surface used by Qwen3TTS."""

    def __init__(self, wavs_per_call: int = 3, samples_per_wav: int = 7680) -> None:
        self.synth_calls: list[str] = []
        self._wavs_per_call = wavs_per_call
        self._samples_per_wav = samples_per_wav

    def synthesize(self, text: str) -> Iterable[np.ndarray]:
        self.synth_calls.append(text)
        for _ in range(self._wavs_per_call):
            yield np.zeros(self._samples_per_wav, dtype=np.float32)


class TestQwen3TTSInit:
    def test_defaults(self, settings):
        tts = Qwen3TTS(settings)

        assert tts.sample_rate == 16000
        assert tts.sample_width == 2
        assert tts.channels == 1
        assert tts._inferencer is None

    def test_custom_sample_rate(self, settings):
        tts = Qwen3TTS(settings, sample_rate=24000)

        assert tts.sample_rate == 24000


class TestSynthesizeIter:
    def test_raises_before_load(self, settings):
        tts = Qwen3TTS(settings)

        with pytest.raises(RuntimeError, match="load"):
            list(tts.synthesize_iter("hello"))

    def test_passes_full_text_to_inferencer(self, settings):
        tts = Qwen3TTS(settings)
        tts._inferencer = FakeInferencer()

        list(tts.synthesize_iter("hello world"))
        list(tts.synthesize_iter("second turn"))

        assert tts._inferencer.synth_calls == ["hello world", "second turn"]

    def test_yields_int16_pcm(self, settings):
        tts = Qwen3TTS(settings, sample_rate=24000)
        tts._inferencer = FakeInferencer(wavs_per_call=2)

        chunks = list(tts.synthesize_iter("hi"))

        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) % 2 == 0  # int16 samples

    def test_resamples_when_target_differs_from_native(self, settings):
        tts = Qwen3TTS(settings, sample_rate=16000)
        tts._inferencer = FakeInferencer(wavs_per_call=1, samples_per_wav=7680)

        chunks = list(tts.synthesize_iter("hi"))

        # 7680 float32 samples @ 24kHz → ~5120 samples @ 16kHz → ~10240 bytes
        assert chunks
        assert len(chunks[0]) < 7680 * 2

    def test_skips_empty_wav_arrays(self, settings):
        tts = Qwen3TTS(settings)
        fake = FakeInferencer()

        def synth(text: str):
            fake.synth_calls.append(text)
            yield np.zeros(0, dtype=np.float32)
            yield np.zeros(7680, dtype=np.float32)

        fake.synthesize = synth
        tts._inferencer = fake

        chunks = list(tts.synthesize_iter("hi"))

        assert len(chunks) == 1


class TestProviderSelection:
    def test_cpu_only_default(self, settings):
        with patch(
            "ovi_voice_assistant.tts.qwen3_tts.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ):
            providers = Qwen3TTS._select_providers()

        assert providers == ["CPUExecutionProvider"]

    def test_coreml_intentionally_skipped(self, settings):
        with patch(
            "ovi_voice_assistant.tts.qwen3_tts.ort.get_available_providers",
            return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ):
            providers = Qwen3TTS._select_providers()

        assert "CoreMLExecutionProvider" not in providers
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_preferred_when_available(self, settings):
        with patch(
            "ovi_voice_assistant.tts.qwen3_tts.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            providers = Qwen3TTS._select_providers()

        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
