"""Tests for ovi_voice_assistant.transport.wifi."""

import struct
from unittest.mock import MagicMock

import pytest

from ovi_voice_assistant.transport import EventType
from ovi_voice_assistant.transport.wifi import (
    MIC_AUDIO_TYPE,
    SPEAKER_AUDIO_TYPE,
    WiFiTransport,
)


class TestWiFiTransportInit:
    def test_default_port(self):
        t = WiFiTransport("192.168.1.42")

        assert t._port == 6055

    def test_custom_port(self):
        t = WiFiTransport("192.168.1.42", port=9999)

        assert t._port == 9999

    def test_is_connected_starts_false(self):
        t = WiFiTransport("192.168.1.42")

        assert t.is_connected is False

    def test_callbacks_start_none(self):
        t = WiFiTransport("192.168.1.42")

        assert t._event_cb is None
        assert t._audio_cb is None
        assert t._disconnect_cb is None
        assert t._connect_cb is None


class TestWiFiTransportCallbacks:
    def test_set_event_callback(self):
        t = WiFiTransport("host")
        cb = MagicMock()

        t.set_event_callback(cb)

        assert t._event_cb is cb

    def test_set_audio_callback(self):
        t = WiFiTransport("host")
        cb = MagicMock()

        t.set_audio_callback(cb)

        assert t._audio_cb is cb

    def test_set_disconnect_callback(self):
        t = WiFiTransport("host")
        cb = MagicMock()

        t.set_disconnect_callback(cb)

        assert t._disconnect_cb is cb

    def test_set_connect_callback(self):
        t = WiFiTransport("host")
        cb = MagicMock()

        t.set_connect_callback(cb)

        assert t._connect_cb is cb


class TestSendFrame:
    def _make_transport_with_writer(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True
        return t, writer

    def test_encodes_length_prefix_le(self):
        t, writer = self._make_transport_with_writer()
        data = b"\x01\x02\x03"

        t._send_frame(data)

        expected = struct.pack("<H", 3) + data
        writer.write.assert_called_once_with(expected)

    def test_no_write_when_writer_is_none(self):
        t = WiFiTransport("host")
        assert t._writer is None

        t._send_frame(b"\x01")  # should not raise

    def test_empty_payload(self):
        t, writer = self._make_transport_with_writer()

        t._send_frame(b"")

        expected = struct.pack("<H", 0)
        writer.write.assert_called_once_with(expected)


class TestSendEvent:
    @pytest.mark.asyncio
    async def test_send_event_not_connected_does_nothing(self):
        t = WiFiTransport("host")
        assert not t.is_connected

        await t.send_event(EventType.TTS_START)  # should not raise

    @pytest.mark.asyncio
    async def test_send_event_connected_writes_correct_frame(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True
        payload = b"\xaa\xbb"

        await t.send_event(EventType.VAD_START, payload)

        inner = bytes([int(EventType.VAD_START)]) + payload
        expected = struct.pack("<H", len(inner)) + inner
        writer.write.assert_called_once_with(expected)

    @pytest.mark.asyncio
    async def test_send_event_no_payload(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True

        await t.send_event(EventType.MIC_STOP)

        inner = bytes([int(EventType.MIC_STOP)])
        expected = struct.pack("<H", len(inner)) + inner
        writer.write.assert_called_once_with(expected)


class TestSendAudio:
    @pytest.mark.asyncio
    async def test_send_audio_not_connected_does_nothing(self):
        t = WiFiTransport("host")
        assert not t.is_connected

        await t.send_audio(b"\x00" * 160)  # should not raise

    @pytest.mark.asyncio
    async def test_send_audio_connected_writes_correct_frame(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True
        audio = bytes(range(10))

        await t.send_audio(audio)

        inner = bytes([SPEAKER_AUDIO_TYPE]) + audio
        expected = struct.pack("<H", len(inner)) + inner
        writer.write.assert_called_once_with(expected)


class TestFrameFormat:
    def test_control_event_frame_layout(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True
        payload = b"\x01\x02\x03\x04"

        t._send_frame(bytes([int(EventType.ERROR)]) + payload)

        raw = writer.write.call_args[0][0]
        length = struct.unpack("<H", raw[:2])[0]
        assert length == 1 + len(payload)
        assert raw[2] == int(EventType.ERROR)
        assert raw[3:] == payload

    def test_speaker_audio_frame_layout(self):
        t = WiFiTransport("host")
        writer = MagicMock()
        t._writer = writer
        t._connected = True
        audio = b"\xff" * 20

        t._send_frame(bytes([SPEAKER_AUDIO_TYPE]) + audio)

        raw = writer.write.call_args[0][0]
        length = struct.unpack("<H", raw[:2])[0]
        assert length == 1 + len(audio)
        assert raw[2] == SPEAKER_AUDIO_TYPE
        assert raw[3:] == audio

    def test_mic_audio_type_constant(self):
        result = MIC_AUDIO_TYPE

        assert result == 0x20

    def test_speaker_audio_type_constant(self):
        result = SPEAKER_AUDIO_TYPE

        assert result == 0x21
