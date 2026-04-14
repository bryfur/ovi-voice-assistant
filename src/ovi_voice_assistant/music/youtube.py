"""YouTube Music search provider."""

from __future__ import annotations

import asyncio
import logging

from ovi_voice_assistant.music.music_player import MusicTrack

logger = logging.getLogger(__name__)


async def search_youtube(query: str, limit: int = 20) -> list[MusicTrack]:
    """Search YouTube Music and return tracks with video IDs for streaming."""
    from ytmusicapi import YTMusic

    def _search() -> list[MusicTrack]:
        yt = YTMusic()
        results = yt.search(query, filter="songs", limit=limit)
        tracks: list[MusicTrack] = []
        for item in results:
            video_id = item.get("videoId")
            if not video_id:
                continue
            artists = ", ".join(
                a["name"] for a in item.get("artists", []) if "name" in a
            )
            album = ""
            if item.get("album") and item["album"].get("name"):
                album = item["album"]["name"]
            duration = item.get("duration_seconds", 0)
            tracks.append(
                MusicTrack(
                    title=item.get("title", ""),
                    artist=artists,
                    album=album,
                    duration_seconds=duration,
                    video_id=video_id,
                    service="youtube",
                )
            )
        return tracks

    tracks = await asyncio.to_thread(_search)
    logger.info("YouTube Music search %r → %d results", query, len(tracks))
    return tracks
