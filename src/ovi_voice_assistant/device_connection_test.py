"""Tests for DeviceConnection helpers: _EncodingOutput."""

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovi_voice_assistant.codec.pcm import PcmCodec
from ovi_voice_assistant.device_connection import _EncodingOutput
from ovi_voice_assistant.transport import EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_transport() -> MagicMock:
    transport = MagicMock()
    transport.send_event = AsyncMock()
    transport.send_audio = AsyncMock()
    return transport


# ---------------------------------------------------------------------------
# _EncodingOutput.send_event
# ---------------------------------------------------------------------------


class TestEncodingOutputSendEvent:
    @pytest.mark.asyncio
    async def test_tts_start_sends_audio_config_first(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        await output.send_event(EventType.TTS_START)

        assert transport.send_event.call_count == 2
        # First call: AUDIO_CONFIG
        first_call = transport.send_event.call_args_list[0]
        assert first_call[0][0] == EventType.AUDIO_CONFIG
        # Verify the config payload is correctly packed
        payload = first_call[0][1]
        rate, frame_bytes, codec_id, channels = struct.unpack("<IHBB", payload)
        assert rate == 16000
        assert frame_bytes == codec.encoded_frame_bytes
        assert codec_id == 0  # PCM
        assert channels == 1
        # Second call: TTS_START
        second_call = transport.send_event.call_args_list[1]
        assert second_call[0][0] == EventType.TTS_START

    @pytest.mark.asyncio
    async def test_non_tts_start_forwarded_directly(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        await output.send_event(EventType.TTS_END, b"\x01")

        transport.send_event.assert_awaited_once_with(EventType.TTS_END, b"\x01")

    @pytest.mark.asyncio
    async def test_other_events_no_audio_config(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        for ev in (EventType.VAD_START, EventType.MIC_STOP, EventType.ERROR):
            transport.send_event.reset_mock()

            await output.send_event(ev)

            assert transport.send_event.call_count == 1
            assert transport.send_event.call_args[0][0] == ev


# ---------------------------------------------------------------------------
# _EncodingOutput.send_audio
# ---------------------------------------------------------------------------


class TestEncodingOutputSendAudio:
    @pytest.mark.asyncio
    async def test_buffers_until_full_frame(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)  # pcm_frame_bytes = 640
        output = _EncodingOutput(transport, codec)

        # Send less than one frame
        await output.send_audio(b"\x00" * 320)
        transport.send_audio.assert_not_awaited()

        # Send the rest to complete exactly one frame
        await output.send_audio(b"\x00" * 320)

        transport.send_audio.assert_awaited_once()
        # PcmCodec is passthrough, so encoded data == pcm frame
        assert len(transport.send_audio.call_args[0][0]) == 640

    @pytest.mark.asyncio
    async def test_multiple_frames_sent(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)  # pcm_frame_bytes = 640
        output = _EncodingOutput(transport, codec)

        await output.send_audio(b"\x00" * 1280)

        assert transport.send_audio.call_count == 2

    @pytest.mark.asyncio
    async def test_leftover_stays_buffered(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        # 700 bytes = 1 frame (640) + 60 leftover
        await output.send_audio(b"\x00" * 700)
        assert transport.send_audio.call_count == 1

        # Internal buffer should have 60 bytes; send 580 more to complete
        await output.send_audio(b"\x00" * 580)

        assert transport.send_audio.call_count == 2

    @pytest.mark.asyncio
    async def test_encoded_data_matches_input(self):
        """PcmCodec is passthrough, so encoded frame should equal the PCM input."""
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)
        frame = bytes(range(256)) * 2 + bytes(range(128))  # 640 bytes
        assert len(frame) == 640

        await output.send_audio(frame)

        transport.send_audio.assert_awaited_once_with(frame)


# ---------------------------------------------------------------------------
# _EncodingOutput.flush
# ---------------------------------------------------------------------------


class TestEncodingOutputFlush:
    @pytest.mark.asyncio
    async def test_flush_pads_and_sends(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)  # pcm_frame_bytes = 640
        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        transport.send_audio.assert_not_awaited()

        await output.flush()

        transport.send_audio.assert_awaited_once()
        sent = transport.send_audio.call_args[0][0]
        assert len(sent) == 640
        # First 100 bytes are \x01, rest is zero-padded
        assert sent[:100] == b"\x01" * 100
        assert sent[100:] == b"\x00" * 540

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_no_send(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        await output.flush()

        transport.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        await output.flush()
        transport.send_audio.reset_mock()

        await output.flush()

        transport.send_audio.assert_not_awaited()


# ---------------------------------------------------------------------------
# _EncodingOutput.reset
# ---------------------------------------------------------------------------


class TestEncodingOutputReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x00" * 640)
        assert output._frame_count == 1
        assert output._t0 != 0.0
        await output.send_audio(b"\x01" * 100)

        output.reset()

        assert output._pcm_buf == b""
        assert output._frame_count == 0
        assert output._t0 == 0.0

    @pytest.mark.asyncio
    async def test_reset_discards_buffered_audio(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        transport.send_audio.assert_not_awaited()

        output.reset()
        await output.flush()

        transport.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reset_then_flush_no_send(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)

        output.reset()
        await output.flush()

        transport.send_audio.assert_not_awaited()
