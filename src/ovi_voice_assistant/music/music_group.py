"""Multi-device synchronized music playback via SNTP clock sync.

All devices sync their clocks via NTP. The server sends audio to every device
simultaneously, then sends a SYNC_PLAY event with a future NTP timestamp.
Each device buffers audio and starts playback at the same wall-clock moment.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import TYPE_CHECKING

from ovi_voice_assistant.music.music_player import MusicPlayer, MusicTrack
from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.transport import EventType

if TYPE_CHECKING:
    from ovi_voice_assistant.device_connection import _EncodingOutput
    from ovi_voice_assistant.music.browser_music import BrowserMusic

logger = logging.getLogger(__name__)

# How far in the future to schedule playback start (ms).
# Must be large enough for audio to reach all devices and buffer.
SYNC_BUFFER_MS = 500


class _FanoutOutput(PipelineOutput):
    """Sends the same PCM and events to multiple encoding outputs."""

    def __init__(self, outputs: list[_EncodingOutput]) -> None:
        self._outputs = outputs

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        await asyncio.gather(*(o.send_event(event, payload) for o in self._outputs))

    async def send_audio(self, pcm: bytes) -> None:
        await asyncio.gather(*(o.send_audio(pcm) for o in self._outputs))


class MusicGroup:
    """Coordinates music playback across multiple devices.

    Owns a single MusicPlayer (one ffmpeg process) and fans out
    PCM audio to all grouped device outputs simultaneously.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        browsers: dict[str, BrowserMusic] | None = None,
    ) -> None:
        self.player = MusicPlayer(
            sample_rate=sample_rate, channels=channels, browsers=browsers
        )
        self._outputs: list[_EncodingOutput] = []
        self._transports: list = []  # DeviceTransport references for SYNC_PLAY
        self._task: asyncio.Task | None = None
        self._sample_rate = sample_rate

    def add_device(self, output: _EncodingOutput, transport) -> None:
        """Register a device's music output and transport."""
        self._outputs.append(output)
        self._transports.append(transport)

    async def play(self, tracks: list[MusicTrack]) -> None:
        """Queue tracks for playback. Streaming starts after the voice pipeline
        finishes (via ``DeviceConnection._run``), so TTS and music never
        compete for the same decoder."""
        await self.stop()
        self.player.set_queue(tracks)

    async def pause(self) -> None:
        """Pause playback on all devices."""
        self.player.pause()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError, Exception:
                pass
        # Signal all devices to stop speaker
        for out in self._outputs:
            try:
                await out.flush()
                await out.send_event(EventType.TTS_END)
            except Exception:
                pass

    async def resume(self) -> None:
        """Resume synchronized playback on all devices."""
        self.player.resume()
        if self.player.is_active:
            self._task = asyncio.create_task(self._stream_all())

    def skip(self) -> MusicTrack | None:
        """Skip to next track. Restarts streaming."""
        track = self.player.skip()
        if track and self._task and not self._task.done():
            self._task.cancel()
            self._task = asyncio.create_task(self._stream_all())
        return track

    async def stop(self) -> None:
        """Stop playback and clear queue on all devices."""
        self.player.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError, Exception:
                pass
        for out in self._outputs:
            try:
                await out.flush()
                await out.send_event(EventType.TTS_END)
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        return self.player.is_active

    async def _stream_all(self) -> None:
        """Stream music to all devices with synchronized start."""
        fanout = _FanoutOutput(self._outputs)

        # Reset all outputs
        for out in self._outputs:
            await out.reset()

        # Send TTS_START (which also sends AUDIO_CONFIG) to all devices
        await fanout.send_event(EventType.TTS_START)

        # Send SYNC_PLAY with a future NTP timestamp to all devices
        target_ms = int(time.time() * 1000) + SYNC_BUFFER_MS
        sync_payload = struct.pack("<Q", target_ms)
        for transport in self._transports:
            try:
                await transport.send_event(EventType.SYNC_PLAY, sync_payload)
            except Exception:
                logger.exception("Failed to send SYNC_PLAY")

        logger.info(
            "Sync playback scheduled: target=%d ms, buffer=%d ms, devices=%d",
            target_ms,
            SYNC_BUFFER_MS,
            len(self._outputs),
        )

        try:
            await self.player.stream(fanout)
            # Normal completion — flush and end
            for out in self._outputs:
                await out.flush()
            await fanout.send_event(EventType.TTS_END)
        except asyncio.CancelledError:
            for out in self._outputs:
                try:
                    await out.flush()
                    await out.send_event(EventType.TTS_END)
                except Exception:
                    pass
            raise
