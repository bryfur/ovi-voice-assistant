"""Spotify playback via browser audio capture."""

from __future__ import annotations

import logging

from ovi_voice_assistant.music.browser_music import BrowserMusic
from ovi_voice_assistant.music.music_player import MusicTrack

logger = logging.getLogger(__name__)

# Grab an access token from Spotify's internal endpoint, then hit the Web API.
_SEARCH_JS = """\
async ({ query, limit }) => {
  const tokenRes = await fetch('/get_access_token');
  const { accessToken } = await tokenRes.json();
  const res = await fetch(
    `https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track&limit=${limit}`,
    { headers: { Authorization: `Bearer ${accessToken}` } },
  );
  const data = await res.json();
  return (data.tracks?.items || []).map(t => ({
    id: t.id,
    uri: t.uri,
    title: t.name,
    artist: t.artists.map(a => a.name).join(', '),
    album: t.album?.name || '',
    duration: Math.round(t.duration_ms / 1000),
  }));
}
"""

# Play a track via Spotify Connect Web API, then poll until it finishes.
_PLAY_JS = """\
async ({ uri, durationSec }) => {
  const tokenRes = await fetch('/get_access_token');
  const { accessToken } = await tokenRes.json();
  const headers = { Authorization: `Bearer ${accessToken}` };

  // Find the web player device
  const devRes = await fetch(
    'https://api.spotify.com/v1/me/player/devices', { headers },
  );
  const { devices } = await devRes.json();
  const device = devices?.find(d => d.type === 'Computer');
  const qs = device ? `?device_id=${device.id}` : '';

  // Start playback
  await fetch(`https://api.spotify.com/v1/me/player/play${qs}`, {
    method: 'PUT',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ uris: [uri] }),
  });

  // Poll until the track ends
  await new Promise(resolve => {
    let elapsed = 0;
    const iv = setInterval(async () => {
      elapsed += 2;
      try {
        const r = await fetch(
          'https://api.spotify.com/v1/me/player', { headers },
        );
        if (!r.ok) return;
        const s = await r.json();
        if (!s.is_playing) { clearInterval(iv); resolve(); }
      } catch {}
      // Safety: resolve after 1.5x the reported duration
      if (durationSec && elapsed > durationSec * 1.5) {
        clearInterval(iv); resolve();
      }
    }, 2000);
  });
}
"""


class SpotifyMusic(BrowserMusic):
    """Spotify via open.spotify.com."""

    _URL = "https://open.spotify.com"
    _PROFILE_NAME = "spotify-profile"

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
                service="spotify",
            )
            for r in results
        ]

    async def _play_and_wait(self, track: MusicTrack) -> None:
        uri = f"spotify:track:{track.song_id}"
        await self._page.evaluate(
            _PLAY_JS, {"uri": uri, "durationSec": track.duration_seconds}
        )

    async def stop_playback(self) -> None:
        if self._page:
            await self._page.evaluate("""\
                async () => {
                  const r = await fetch('/get_access_token');
                  const { accessToken } = await r.json();
                  await fetch('https://api.spotify.com/v1/me/player/pause', {
                    method: 'PUT',
                    headers: { Authorization: `Bearer ${accessToken}` },
                  });
                }
            """)
