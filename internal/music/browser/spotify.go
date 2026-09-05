package browser

import "github.com/bryfur/ovi-voice-assistant/internal/music"

// Spotify plays through open.spotify.com, using the page's own access
// token against the Web API.
func Spotify() *Provider {
	return &Provider{
		Session:  newSession("https://open.spotify.com", "spotify-profile", spotifyStopJS),
		name:     "spotify",
		searchJS: spotifySearchJS,
		playJS:   spotifyPlayJS,
		playArg: func(t music.Track) any {
			return map[string]any{"uri": "spotify:track:" + t.ID, "durationSec": t.Duration}
		},
	}
}

const spotifySearchJS = `
async ({ query, limit }) => {
  const { accessToken } = await (await fetch('/get_access_token')).json();
  const res = await fetch(
    'https://api.spotify.com/v1/search?q=' + encodeURIComponent(query) + '&type=track&limit=' + limit,
    { headers: { Authorization: 'Bearer ' + accessToken } });
  const data = await res.json();
  return (data.tracks?.items || []).map(t => ({
    id: t.id,
    title: t.name,
    artist: t.artists.map(a => a.name).join(', '),
    album: t.album?.name || '',
    duration: Math.round(t.duration_ms / 1000),
  }));
}
`

// Start the track on the web player, then poll until it stops.
const spotifyPlayJS = `
async ({ uri, durationSec }) => {
  const { accessToken } = await (await fetch('/get_access_token')).json();
  const headers = { Authorization: 'Bearer ' + accessToken };
  const { devices } = await (await fetch('https://api.spotify.com/v1/me/player/devices', { headers })).json();
  const device = devices?.find(d => d.type === 'Computer');
  await fetch('https://api.spotify.com/v1/me/player/play' + (device ? '?device_id=' + device.id : ''), {
    method: 'PUT',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify({ uris: [uri] }),
  });
  await new Promise(resolve => {
    let elapsed = 0;
    const iv = setInterval(async () => {
      elapsed += 2;
      try {
        const r = await fetch('https://api.spotify.com/v1/me/player', { headers });
        if (r.ok && !(await r.json()).is_playing) { clearInterval(iv); resolve(); }
      } catch {}
      if (durationSec && elapsed > durationSec * 1.5) { clearInterval(iv); resolve(); }
    }, 2000);
  });
  return true;
}
`

const spotifyStopJS = `
async () => {
  const { accessToken } = await (await fetch('/get_access_token')).json();
  await fetch('https://api.spotify.com/v1/me/player/pause', {
    method: 'PUT', headers: { Authorization: 'Bearer ' + accessToken },
  });
  return true;
}
`
