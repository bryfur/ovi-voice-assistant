"""Tests for ovi_voice_assistant.codec.pcm."""

import pytest

from ovi_voice_assistant.codec import CodecType
from ovi_voice_assistant.codec.pcm import PcmCodec


class TestPcmCodec:
    @pytest.fixture()
    def codec(self):
        return PcmCodec(16000, channels=1)

    def test_codec_id(self, codec):
        assert codec.codec_id == 0

    def test_codec_type(self, codec):
        assert codec.codec_type == CodecType.PCM

    def test_sample_rate(self, codec):
        assert codec.sample_rate == 16000

    def test_frame_duration_ms(self, codec):
        assert codec.frame_duration_ms == 20

    def test_pcm_frame_bytes_16k_mono(self, codec):
        # 16000 * 1 * 2 * 20 / 1000 = 640
        assert codec.pcm_frame_bytes == 640

    def test_encoded_frame_bytes_equals_pcm(self, codec):
        assert codec.encoded_frame_bytes == codec.pcm_frame_bytes

    def test_pcm_frame_bytes_48k_stereo(self):
        c = PcmCodec(48000, channels=2)

        # 48000 * 2 * 2 * 20 / 1000 = 3840
        assert c.pcm_frame_bytes == 3840

    def test_encode_is_identity(self, codec):
        data = b"\x01\x02\x03\x04"

        result = codec.encode(data)

        assert result is data

    def test_decode_is_identity(self, codec):
        data = b"\xaa\xbb\xcc\xdd"

        result = codec.decode(data)

        assert result is data
