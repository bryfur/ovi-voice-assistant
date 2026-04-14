"""Music streaming — search, queue, and stream audio to devices."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ovi_voice_assistant.music.music_group import MusicGroup
from ovi_voice_assistant.music.music_player import MusicPlayer, MusicTrack

if TYPE_CHECKING:
    from ovi_voice_assistant.music.browser_music import BrowserMusic

__all__ = [
    "MusicGroup",
    "MusicPlayer",
    "MusicTrack",
    "register_browser",
    "resolve_youtube_id",
    "search_music",
]

# Browser-based music providers keyed by service name ("apple", "spotify").
_browsers: dict[str, BrowserMusic] = {}


def register_browser(service: str, browser: BrowserMusic) -> None:
    """Register a browser music provider (e.g. ``register_browser("apple", apple)``)."""
    _browsers[service] = browser


def get_browsers() -> dict[str, BrowserMusic]:
    """Return registered browser providers (for passing to MusicPlayer/MusicGroup)."""
    return _browsers


async def resolve_youtube_id(search_query: str) -> str:
    """Find the best YouTube Music video ID for a search query."""
    from ytmusicapi import YTMusic

    def _search() -> str:
        yt = YTMusic()
        results = yt.search(search_query, filter="songs", limit=1)
        if results and results[0].get("videoId"):
            return results[0]["videoId"]
        return ""

    return await asyncio.to_thread(_search)


async def search_music(query: str, service: str = "youtube") -> list[MusicTrack]:
    """Search for music. *service* selects the provider."""
    browser = _browsers.get(service)
    if browser:
        return await browser.search(query)

    if service == "youtube":
        from ovi_voice_assistant.music.youtube import search_youtube

        return await search_youtube(query)

    available = ["youtube", *_browsers.keys()]
    raise ValueError(f"Unknown music service {service!r} (available: {available})")
