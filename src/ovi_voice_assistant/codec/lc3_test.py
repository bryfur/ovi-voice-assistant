"""Tests for ovi_voice_assistant.codec.lc3."""

import pytest

from ovi_voice_assistant.codec import CodecType
from ovi_voice_assistant.codec.lc3 import Lc3Codec


class TestLc3Codec:
    @pytest.fixture()
    def codec(self):
        return Lc3Codec(16000, channels=1)

    def test_codec_id(self, codec):
        assert codec.codec_id == 1

    def test_codec_type(self, codec):
        assert codec.codec_type == CodecType.LC3

    def test_sample_rate(self, codec):
        assert codec.sample_rate == 16000

    def test_frame_duration_ms(self, codec):
        assert codec.frame_duration_ms == 10

    def test_encoded_frame_bytes(self, codec):
        assert codec.encoded_frame_bytes == 40

    def test_pcm_frame_bytes(self, codec):
        # 10ms at 16000Hz mono: 160 samples * 2 bytes = 320
        assert codec.pcm_frame_bytes == 160 * 1 * 2

    def test_encode_produces_bytes(self, codec):
        pcm = bytes(codec.pcm_frame_bytes)

        encoded = codec.encode(pcm)

        assert isinstance(encoded, bytes)
        assert len(encoded) == codec.encoded_frame_bytes

    def test_round_trip_preserves_length(self, codec):
        pcm = bytes(codec.pcm_frame_bytes)
        encoded = codec.encode(pcm)

        decoded = codec.decode(encoded)

        assert len(decoded) == codec.pcm_frame_bytes
