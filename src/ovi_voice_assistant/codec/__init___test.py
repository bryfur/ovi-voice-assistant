"""Tests for ovi_voice_assistant.codec (factory, CodecType, rate snapping)."""

import pytest

from ovi_voice_assistant.codec import (
    LC3_VALID_RATES,
    OPUS_VALID_RATES,
    CodecType,
    _nearest_valid_rate,
    create_codec,
)
from ovi_voice_assistant.codec.pcm import PcmCodec


class TestCodecType:
    def test_values(self):
        assert CodecType.PCM.value == "pcm"
        assert CodecType.LC3.value == "lc3"
        assert CodecType.OPUS.value == "opus"

    def test_from_string(self):
        assert CodecType("pcm") is CodecType.PCM
        assert CodecType("lc3") is CodecType.LC3
        assert CodecType("opus") is CodecType.OPUS

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            CodecType("flac")


class TestNearestValidRate:
    def test_exact_match(self):
        result = _nearest_valid_rate(16000, LC3_VALID_RATES)

        assert result == 16000

    def test_nearest_below(self):
        result = _nearest_valid_rate(15000, LC3_VALID_RATES)

        assert result == 16000

    def test_nearest_above(self):
        result = _nearest_valid_rate(9000, LC3_VALID_RATES)

        assert result == 8000

    def test_midpoint_picks_lower(self):
        result = _nearest_valid_rate(12000, LC3_VALID_RATES)

        assert result == 8000

    def test_very_low(self):
        result = _nearest_valid_rate(1, LC3_VALID_RATES)

        assert result == 8000

    def test_very_high(self):
        result = _nearest_valid_rate(96000, LC3_VALID_RATES)

        assert result == 48000

    def test_opus_rates(self):
        result_16k = _nearest_valid_rate(16000, OPUS_VALID_RATES)
        result_11k = _nearest_valid_rate(11000, OPUS_VALID_RATES)
        result_44k = _nearest_valid_rate(44100, OPUS_VALID_RATES)

        assert result_16k == 16000
        assert result_11k == 12000
        assert result_44k == 48000


class TestCreateCodec:
    def test_pcm_any_rate(self):
        codec = create_codec(CodecType.PCM, 44100)

        assert isinstance(codec, PcmCodec)
        assert codec.sample_rate == 44100

    def test_pcm_string_type(self):
        codec = create_codec("pcm", 16000)

        assert isinstance(codec, PcmCodec)

    def test_lc3_valid_rate(self):
        codec = create_codec(CodecType.LC3, 16000)

        assert codec.codec_type == CodecType.LC3
        assert codec.sample_rate == 16000

    def test_lc3_invalid_rate_snaps(self):
        codec = create_codec(CodecType.LC3, 15000)

        assert codec.sample_rate == 16000

    def test_lc3_string_type(self):
        codec = create_codec("lc3", 16000)

        assert codec.codec_type == CodecType.LC3

    def test_opus_valid_rate(self):
        codec = create_codec(CodecType.OPUS, 16000)

        assert codec.codec_type == CodecType.OPUS
        assert codec.sample_rate == 16000

    def test_opus_invalid_rate_snaps(self):
        codec = create_codec(CodecType.OPUS, 44100)

        assert codec.sample_rate == 48000

    def test_opus_string_type(self):
        codec = create_codec("opus", 16000)

        assert codec.codec_type == CodecType.OPUS

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError):
            create_codec("flac", 16000)
