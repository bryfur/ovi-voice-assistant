"""Tests for DeviceConnection helpers: _EncodingOutput."""

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovi_voice_assistant.codec.pcm import PcmCodec
from ovi_voice_assistant.device_connection import _EncodingOutput
from ovi_voice_assistant.transport import EventType


def _mock_transport() -> MagicMock:
    transport = MagicMock()
    transport.send_event = AsyncMock()
    transport.send_audio = AsyncMock()
    return transport


class TestEncodingOutputSendEvent:
    @pytest.mark.asyncio
    async def test_tts_start_sends_audio_config_first(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        output = _EncodingOutput(transport, codec)

        await output.send_event(EventType.TTS_START)

        assert transport.send_event.call_count == 2
        first_call = transport.send_event.call_args_list[0]
        assert first_call[0][0] == EventType.AUDIO_CONFIG
        payload = first_call[0][1]
        rate, frame_bytes, codec_id, channels = struct.unpack("<IHBB", payload)
        assert rate == 16000
        assert frame_bytes == codec.encoded_frame_bytes
        assert codec_id == 0
        assert channels == 1
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


class TestEncodingOutputSendAudio:
    @pytest.mark.asyncio
    async def test_buffers_until_full_frame(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x00" * 320)
        await output.flush()

        assert len(transport.send_audio.call_args[0][0]) == 640

    @pytest.mark.asyncio
    async def test_multiple_frames_sent(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x00" * 1280)
        await output.flush()

        assert transport.send_audio.call_count == 2

    @pytest.mark.asyncio
    async def test_leftover_stays_buffered(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x00" * 700)
        await output.send_audio(b"\x00" * 580)
        await output.flush()

        assert transport.send_audio.call_count == 2

    @pytest.mark.asyncio
    async def test_encoded_data_matches_input(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)
        frame = bytes(range(256)) * 2 + bytes(range(128))
        assert len(frame) == 640

        output = _EncodingOutput(transport, codec)
        await output.send_audio(frame)
        await output.flush()

        transport.send_audio.assert_awaited_once_with(frame)


class TestEncodingOutputFlush:
    @pytest.mark.asyncio
    async def test_flush_pads_and_sends(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        await output.flush()

        transport.send_audio.assert_awaited_once()
        sent = transport.send_audio.call_args[0][0]
        assert len(sent) == 640
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


class TestEncodingOutputReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x00" * 640)
        await output.flush()
        await output.send_audio(b"\x01" * 100)
        await output.reset()

        assert output._pcm_buf == b""
        assert output._frame_count == 0
        assert output._t0 == 0.0

    @pytest.mark.asyncio
    async def test_reset_discards_buffered_audio(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        await output.reset()
        await output.flush()

        transport.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reset_then_flush_no_send(self):
        transport = _mock_transport()
        codec = PcmCodec(16000)

        output = _EncodingOutput(transport, codec)
        await output.send_audio(b"\x01" * 100)
        await output.reset()
        await output.flush()

        transport.send_audio.assert_not_awaited()
