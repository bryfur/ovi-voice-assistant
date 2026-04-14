"""Tests for ovi_voice_assistant.transport (EventType, AudioConfig)."""

import pytest

from ovi_voice_assistant.transport import AudioConfig, EventType


class TestEventType:
    """Verify all EventType members have the expected integer values."""

    @pytest.mark.parametrize(
        "member, expected",
        [
            (EventType.WAKE_WORD, 0x01),
            (EventType.VAD_START, 0x02),
            (EventType.MIC_STOP, 0x03),
            (EventType.TTS_START, 0x04),
            (EventType.TTS_END, 0x05),
            (EventType.CONTINUE, 0x06),
            (EventType.ERROR, 0x07),
            (EventType.AUDIO_CONFIG, 0x08),
            (EventType.MIC_CONFIG, 0x09),
            (EventType.WAKE_ABORT, 0x0A),
        ],
    )
    def test_event_values(self, member, expected):
        result = int(member)

        assert result == expected

    def test_event_count(self):
        result = len(EventType)

        assert result == 11


class TestAudioConfig:
    def test_construction_and_fields(self):
        cfg = AudioConfig(sample_rate=16000, encoded_frame_bytes=160, codec_type=1)

        assert cfg.sample_rate == 16000
        assert cfg.encoded_frame_bytes == 160
        assert cfg.codec_type == 1

    def test_pcm_defaults(self):
        cfg = AudioConfig(sample_rate=16000, encoded_frame_bytes=0, codec_type=0)

        assert cfg.codec_type == 0
        assert cfg.encoded_frame_bytes == 0
