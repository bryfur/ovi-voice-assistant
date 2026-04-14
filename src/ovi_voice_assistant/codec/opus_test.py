"""Tests for ovi_voice_assistant.codec.opus."""

import pytest

from ovi_voice_assistant.codec import CodecType
from ovi_voice_assistant.codec.opus import OpusCodec


class TestOpusCodec:
    @pytest.fixture()
    def codec(self):
        return OpusCodec(16000, channels=1)

    def test_codec_id(self, codec):
        assert codec.codec_id == 2

    def test_codec_type(self, codec):
        assert codec.codec_type == CodecType.OPUS

    def test_sample_rate(self, codec):
        assert codec.sample_rate == 16000

    def test_frame_duration_ms(self, codec):
        assert codec.frame_duration_ms == 20

    def test_encoded_frame_bytes(self, codec):
        assert codec.encoded_frame_bytes == 80

    def test_pcm_frame_bytes(self, codec):
        # 20ms at 16000Hz mono: 320 samples * 2 bytes = 640
        assert codec.pcm_frame_bytes == 320 * 1 * 2

    def test_encode_produces_bytes(self, codec):
        pcm = bytes(codec.pcm_frame_bytes)

        encoded = codec.encode(pcm)

        assert isinstance(encoded, bytes)
        # VBR: encoded size varies, but should be non-empty
        assert len(encoded) > 0

    def test_round_trip_preserves_length(self, codec):
        pcm = bytes(codec.pcm_frame_bytes)
        encoded = codec.encode(pcm)

        decoded = codec.decode(encoded)

        assert len(decoded) == codec.pcm_frame_bytes
