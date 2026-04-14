"""Apple Music playback via browser audio capture."""

from __future__ import annotations

import asyncio
import logging

from ovi_voice_assistant.music.browser_music import BrowserMusic
from ovi_voice_assistant.music.music_player import MusicTrack

logger = logging.getLogger(__name__)

_SEARCH_JS = """\
async ({ query, limit }) => {
  const mk = window.MusicKit?.getInstance?.();
  if (!mk) throw new Error('MusicKit not available — log in first');
  const res = await mk.api.music(
    `/v1/catalog/${mk.storefrontId}/search`,
    { term: query, types: 'songs', limit },
  );
  return (res.data.results.songs?.data || []).map(s => ({
    id: s.id,
    title: s.attributes.name,
    artist: s.attributes.artistName,
    album: s.attributes.albumName || '',
    duration: Math.round((s.attributes.durationInMillis || 0) / 1000),
  }));
}
"""

_PLAY_JS = """\
(songId) => new Promise(resolve => {
  const mk = MusicKit.getInstance();
  const done = ({ state }) => {
    if (state === MusicKit.PlaybackStates.completed
        || state === MusicKit.PlaybackStates.ended) {
      mk.removeEventListener('playbackStateDidChange', done);
      resolve();
    }
  };
  mk.addEventListener('playbackStateDidChange', done);
  mk.setQueue({ song: songId }).then(() => mk.play());
})
"""


class AppleMusic(BrowserMusic):
    """Apple Music via music.apple.com."""

    _URL = "https://music.apple.com"
    _PROFILE_NAME = "apple-music-profile"

    async def search(self, query: str, limit: int = 20) -> list[MusicTrack]:
        results = await self._page.evaluate(
            _SEARCH_JS, {"query": query, "limit": limit}
        )
        return [
            MusicTrack(
                title=r["title"],
                artist=r["artist"],
                album=r.get("album", ""),
                duration_seconds=r.get("duration", 0),
                song_id=r["id"],
                service="apple",
            )
            for r in results
        ]

    async def _play_and_wait(self, track: MusicTrack) -> None:
        await self._page.evaluate(_PLAY_JS, track.song_id)

    async def stop_playback(self) -> None:
        if self._page:
            await self._page.evaluate("MusicKit.getInstance()?.stop()")

    async def wait_for_login(self) -> None:
        """Poll until MusicKit reports an authorized user."""
        for _ in range(10):
            await asyncio.sleep(1)
            ready = await self._page.evaluate(
                "() => typeof MusicKit !== 'undefined' && !!MusicKit.getInstance?.()"
            )
            if ready:
                break
        else:
            logger.warning("MusicKit not found on page")
            return

        if await self._page.evaluate("() => MusicKit.getInstance().isAuthorized"):
            logger.info("Apple Music: already logged in")
            return

        logger.info("Apple Music: waiting for login — sign in via the browser window")
        while True:
            await asyncio.sleep(2)
            if await self._page.evaluate("() => MusicKit.getInstance().isAuthorized"):
                logger.info("Apple Music: login successful")
                return
