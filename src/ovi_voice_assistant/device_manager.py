"""Manages connections to all configured Voice PE devices."""

import asyncio
import logging
from dataclasses import dataclass

from ovi_voice_assistant.codec import create_codec
from ovi_voice_assistant.config import DeviceConfig, Settings
from ovi_voice_assistant.device_connection import DeviceConnection
from ovi_voice_assistant.music import MusicGroup
from ovi_voice_assistant.scheduler import Scheduler
from ovi_voice_assistant.transport.wifi import WiFiTransport
from ovi_voice_assistant.voice_assistant import VoiceAssistant

logger = logging.getLogger(__name__)

# How long to wait for competing wake events before picking a winner.
ARBITRATION_WINDOW_S = 0.5


@dataclass
class _WakeCandidate:
    connection: DeviceConnection
    score: int
    wake_word: str


class DeviceManager:
    """Creates and manages DeviceConnection instances for WiFi devices."""

    def __init__(
        self,
        devices: list[DeviceConfig],
        settings: Settings,
        pipeline: VoiceAssistant,
        tts_rate: int,
    ) -> None:
        self._connections: list[DeviceConnection] = []
        self._multi_device = len(devices) > 1

        # Wake-word arbitration state
        self._wake_candidates: list[_WakeCandidate] = []
        self._arbitration_handle: asyncio.TimerHandle | None = None

        # Shared music group for synchronized multi-device playback.
        # Created here so all devices share the same player/queue.
        music_codec = create_codec(settings.transport.codec, 48000, channels=2)
        self._music_group = MusicGroup(
            sample_rate=music_codec.sample_rate, channels=music_codec.channels
        )

        # Only use arbitration callback when there are multiple devices.
        on_wake = self._on_wake if self._multi_device else None

        for device in devices:
            transport = WiFiTransport(device.host, device.port, device.encryption_key)
            codec = create_codec(settings.transport.codec, tts_rate)
            self._connections.append(
                DeviceConnection(
                    transport,
                    codec,
                    pipeline,
                    settings,
                    on_wake=on_wake,
                    name=device.host,
                    music_group=self._music_group,
                )
            )

    async def start(self) -> None:
        for conn in self._connections:
            await conn.start()
        if self._multi_device:
            logger.info(
                "Managing %d device(s) — multi-device arbitration enabled (window=%.1fs)",
                len(self._connections),
                ARBITRATION_WINDOW_S,
            )
        else:
            logger.info("Managing 1 device")

    async def stop(self) -> None:
        if self._arbitration_handle is not None:
            self._arbitration_handle.cancel()
        await self._music_group.stop()
        for conn in self._connections:
            await conn.stop()

    async def announce_all(self, text: str) -> None:
        """Announce text on every connected device."""
        for conn in self._connections:
            conn.announce(text)

    def set_scheduler(self, scheduler: Scheduler) -> None:
        """Attach the scheduler to all device connections."""
        for conn in self._connections:
            conn.set_scheduler(scheduler)

    def set_memory(self, memory) -> None:
        """Attach Hindsight memory to all device connections."""
        for conn in self._connections:
            conn.set_memory(memory)

    # -- Wake-word arbitration --

    async def _on_wake(
        self, connection: DeviceConnection, score: int, wake_word: str
    ) -> None:
        """Called by any DeviceConnection when a wake word is detected."""
        candidate = _WakeCandidate(connection, score, wake_word)
        self._wake_candidates.append(candidate)
        logger.info(
            "Wake candidate: %s score=%d wake_word=%r (%d in window)",
            connection.name,
            score,
            wake_word,
            len(self._wake_candidates),
        )

        if len(self._wake_candidates) == 1:
            # First candidate — start the arbitration timer.
            loop = asyncio.get_running_loop()
            self._arbitration_handle = loop.call_later(
                ARBITRATION_WINDOW_S,
                lambda: asyncio.ensure_future(self._resolve_arbitration()),
            )

    async def _resolve_arbitration(self) -> None:
        """Pick the best device and start its pipeline; abort the rest."""
        self._arbitration_handle = None
        candidates = self._wake_candidates
        self._wake_candidates = []

        if not candidates:
            return

        # Sort by score descending — highest audio energy wins.
        candidates.sort(key=lambda c: c.score, reverse=True)
        winner = candidates[0]
        losers = candidates[1:]

        logger.info(
            "Wake arbitration resolved: winner=%s (score=%d) out of %d candidate(s)",
            winner.connection.name,
            winner.score,
            len(candidates),
        )

        await winner.connection.start_pipeline(winner.wake_word)

        for loser in losers:
            logger.info(
                "  loser=%s (score=%d)",
                loser.connection.name,
                loser.score,
            )
            await loser.connection.abort_wake()
