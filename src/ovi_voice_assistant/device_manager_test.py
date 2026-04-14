"""Tests for DeviceManager wake-word arbitration logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.device_manager import DeviceManager, _WakeCandidate


@pytest.fixture
def settings():
    return Settings(_env_file=None, devices="", openai_api_key="test-key")


def _make_mock_connection(name="device"):
    """Create a mock DeviceConnection with async methods."""
    conn = MagicMock()
    conn.start_pipeline = AsyncMock()
    conn.abort_wake = AsyncMock()
    conn.start = AsyncMock()
    conn.stop = AsyncMock()
    conn.name = name
    return conn


def _make_device_config(host="device1.local"):
    """Create a mock DeviceConfig."""
    cfg = MagicMock()
    cfg.host = host
    cfg.port = 6055
    cfg.encryption_key = None
    return cfg


def _make_manager(num_connections=2):
    """Build a DeviceManager with patched __init__ and mock connections."""
    mgr = object.__new__(DeviceManager)
    mgr._connections = [
        _make_mock_connection(f"device{i}") for i in range(num_connections)
    ]
    mgr._multi_device = num_connections > 1
    mgr._wake_candidates = []
    mgr._arbitration_handle = None
    mgr._music_group = MagicMock()
    mgr._music_group.stop = AsyncMock()
    return mgr


# -- _WakeCandidate dataclass --


class TestWakeCandidate:
    def test_creation(self):
        conn = _make_mock_connection()

        candidate = _WakeCandidate(connection=conn, score=42, wake_word="hey_ovi")

        assert candidate.connection is conn
        assert candidate.score == 42
        assert candidate.wake_word == "hey_ovi"


# -- Constructor device mode --


class TestDeviceManagerInit:
    @patch("ovi_voice_assistant.device_manager.DeviceConnection")
    @patch("ovi_voice_assistant.device_manager.WiFiTransport")
    @patch("ovi_voice_assistant.device_manager.create_codec")
    def test_single_device_no_on_wake(
        self, mock_codec, mock_transport, mock_dc, settings
    ):
        """Single device: _multi_device is False and on_wake is None."""
        devices = [_make_device_config("host1")]

        mgr = DeviceManager(devices, settings, MagicMock(), 22050)

        assert mgr._multi_device is False
        _, kwargs = mock_dc.call_args
        assert kwargs["on_wake"] is None

    @patch("ovi_voice_assistant.device_manager.DeviceConnection")
    @patch("ovi_voice_assistant.device_manager.WiFiTransport")
    @patch("ovi_voice_assistant.device_manager.create_codec")
    def test_multi_device_on_wake_set(
        self, mock_codec, mock_transport, mock_dc, settings
    ):
        """Multiple devices: _multi_device is True and on_wake is the callback."""
        devices = [_make_device_config("host1"), _make_device_config("host2")]

        mgr = DeviceManager(devices, settings, MagicMock(), 22050)

        assert mgr._multi_device is True
        _, kwargs = mock_dc.call_args
        assert kwargs["on_wake"] is not None
        assert callable(kwargs["on_wake"])


# -- _on_wake --


class TestOnWake:
    @pytest.mark.asyncio
    async def test_first_candidate_starts_timer(self):
        mgr = _make_manager(2)
        conn = mgr._connections[0]

        await mgr._on_wake(conn, 100, "hey_ovi")

        assert len(mgr._wake_candidates) == 1
        assert mgr._wake_candidates[0].score == 100
        assert mgr._arbitration_handle is not None

    @pytest.mark.asyncio
    async def test_second_candidate_does_not_start_new_timer(self):
        mgr = _make_manager(2)
        await mgr._on_wake(mgr._connections[0], 100, "hey_ovi")
        first_handle = mgr._arbitration_handle

        await mgr._on_wake(mgr._connections[1], 200, "hey_ovi")

        assert len(mgr._wake_candidates) == 2
        assert mgr._arbitration_handle is first_handle


# -- _resolve_arbitration --


class TestResolveArbitration:
    @pytest.mark.asyncio
    async def test_single_candidate_wins(self):
        mgr = _make_manager(2)
        conn = mgr._connections[0]
        mgr._wake_candidates = [_WakeCandidate(conn, 80, "hey_ovi")]

        await mgr._resolve_arbitration()

        conn.start_pipeline.assert_awaited_once_with("hey_ovi")
        assert mgr._wake_candidates == []
        assert mgr._arbitration_handle is None

    @pytest.mark.asyncio
    async def test_highest_score_wins(self):
        mgr = _make_manager(3)
        c0, c1, c2 = mgr._connections
        mgr._wake_candidates = [
            _WakeCandidate(c0, 50, "hey_ovi"),
            _WakeCandidate(c1, 200, "hey_ovi"),
            _WakeCandidate(c2, 100, "hey_ovi"),
        ]

        await mgr._resolve_arbitration()

        c1.start_pipeline.assert_awaited_once_with("hey_ovi")
        c0.start_pipeline.assert_not_awaited()
        c2.start_pipeline.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_losers_get_abort_wake(self):
        mgr = _make_manager(3)
        c0, c1, c2 = mgr._connections
        mgr._wake_candidates = [
            _WakeCandidate(c0, 50, "hey_ovi"),
            _WakeCandidate(c1, 200, "hey_ovi"),
            _WakeCandidate(c2, 100, "hey_ovi"),
        ]

        await mgr._resolve_arbitration()

        c1.abort_wake.assert_not_awaited()
        c0.abort_wake.assert_awaited_once()
        c2.abort_wake.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_candidates_does_nothing(self):
        mgr = _make_manager(2)
        mgr._wake_candidates = []

        await mgr._resolve_arbitration()

        for conn in mgr._connections:
            conn.start_pipeline.assert_not_awaited()
            conn.abort_wake.assert_not_awaited()


# -- stop --


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_arbitration_handle(self):
        mgr = _make_manager(2)
        mock_handle = MagicMock()
        mgr._arbitration_handle = mock_handle

        await mgr.stop()

        mock_handle.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_handle(self):
        mgr = _make_manager(1)
        assert mgr._arbitration_handle is None

        await mgr.stop()

        for conn in mgr._connections:
            conn.stop.assert_awaited_once()
