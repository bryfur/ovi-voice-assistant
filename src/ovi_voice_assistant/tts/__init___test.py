"""Tests for ovi_voice_assistant.tts (resample utility)."""

import numpy as np

from ovi_voice_assistant.tts import resample


class TestResample:
    def test_downsample_length(self):
        """Resampling 22050 -> 16000 produces the expected number of samples."""
        src_rate, dst_rate = 22050, 16000
        n_samples = 22050  # 1 second of audio
        audio = np.zeros(n_samples, dtype=np.int16)

        result = resample(audio, src_rate, dst_rate)

        expected_len = round(n_samples * dst_rate / src_rate)
        # Allow +-1 sample tolerance for rounding
        assert abs(len(result) - expected_len) <= 1

    def test_identity_resample(self):
        """Resampling at the same rate returns the same length output."""
        audio = np.arange(100, dtype=np.int16)

        result = resample(audio, 16000, 16000)

        assert len(result) == len(audio)

    def test_output_dtype_is_int16(self):
        audio = np.zeros(100, dtype=np.int16)

        result = resample(audio, 16000, 8000)

        assert result.dtype == np.int16

    def test_output_values_clipped(self):
        """Extreme input values should be clipped to int16 range after round-trip."""
        audio = np.array([32767, -32768, 0, 32767, -32768], dtype=np.int16)

        result = resample(audio, 16000, 16000)

        assert result.min() >= -32768
        assert result.max() <= 32767
