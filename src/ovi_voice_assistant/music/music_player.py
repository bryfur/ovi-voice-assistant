"""Music player — queue management and audio streaming via yt-dlp + ffmpeg."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ovi_voice_assistant.pipeline_output import PipelineOutput

if TYPE_CHECKING:
    from ovi_voice_assistant.music.browser_music import BrowserMusic

logger = logging.getLogger(__name__)


@dataclass
class MusicTrack:
    """A single music track."""

    title: str
    artist: str
    album: str = ""
    duration_seconds: int = 0
    video_id: str = ""  # YouTube video ID for yt-dlp streaming
    song_id: str = ""  # Apple Music song ID
    service: str = "youtube"  # "youtube" or "apple"


class MusicPlayer:
    """Manages a music queue and streams audio via yt-dlp + ffmpeg.

    The player holds queue state (tracks, position, active flag).
    Actual streaming is driven by ``DeviceConnection`` which calls
    :meth:`stream` and manages task cancellation on wake-word.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        browsers: dict[str, BrowserMusic] | None = None,
    ) -> None:
        self.queue: list[MusicTrack] = []
        self.current_index: int = 0
        self.is_active: bool = False
        self._sample_rate = sample_rate
        self._channels = channels
        self._process: asyncio.subprocess.Process | None = None
        self._browsers = browsers or {}

    def set_queue(self, tracks: list[MusicTrack], start_index: int = 0) -> None:
        """Replace the queue and mark music as active."""
        self.queue = tracks
        self.current_index = start_index
        self.is_active = True

    def pause(self) -> None:
        self.is_active = False

    def resume(self) -> None:
        if self.queue and self.current_index < len(self.queue):
            self.is_active = True

    def skip(self) -> MusicTrack | None:
        """Advance to next track. Returns the new track or None."""
        if self.current_index + 1 < len(self.queue):
            self.current_index += 1
            self.is_active = True
            return self.queue[self.current_index]
        self.is_active = False
        return None

    def stop(self) -> None:
        self.is_active = False
        self.queue = []
        self.current_index = 0

    def get_current(self) -> MusicTrack | None:
        if self.queue and self.current_index < len(self.queue):
            return self.queue[self.current_index]
        return None

    async def stream(self, output: PipelineOutput) -> None:
        """Stream tracks starting at *current_index* until queue ends or cancelled."""
        while self.is_active and self.current_index < len(self.queue):
            track = self.queue[self.current_index]
            logger.info("Streaming: %s — %s", track.artist, track.title)
            await self._stream_track(output, track)
            self.current_index += 1

        # Finished all tracks
        self.is_active = False

    async def _stream_track(self, output: PipelineOutput, track: MusicTrack) -> None:
        """Stream a track via browser capture or yt-dlp + ffmpeg."""
        browser = self._browsers.get(track.service)
        if browser:
            await browser.stream_track(track, output)
            return

        audio_url = await _extract_audio_url(track.video_id)

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "5",
            "-i",
            audio_url,
            "-f",
            "s16le",
            "-ar",
            str(self._sample_rate),
            "-ac",
            str(self._channels),
            "-loglevel",
            "error",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = process

        # Read ~20ms chunks of 16-bit PCM
        chunk_bytes = self._sample_rate * self._channels * 2 * 20 // 1000
        try:
            while True:
                data = await process.stdout.read(chunk_bytes)
                if not data:
                    break
                await output.send_audio(data)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
            self._process = None


async def _extract_audio_url(video_id: str) -> str:
    """Use yt-dlp to get a direct audio stream URL for a YouTube video."""
    import yt_dlp

    def _extract() -> str:
        opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://music.youtube.com/watch?v={video_id}",
                download=False,
            )
            return info["url"]

    return await asyncio.to_thread(_extract)
