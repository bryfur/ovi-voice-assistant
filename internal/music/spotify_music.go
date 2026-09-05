package music

import (
	"context"

	"github.com/bryfur/ovi-voice-assistant/internal/audio"
)

// Grab an access token from Spotify's internal endpoint, then hit the Web API.
const spotifySearchJS = `
async ({ query, limit }) => {
  const tokenRes = await fetch('/get_access_token');
  const { accessToken } = await tokenRes.json();
  const res = await fetch(
    'https://api.spotify.com/v1/search?q=' + encodeURIComponent(query) + '&type=track&limit=' + limit,
    { headers: { Authorization: 'Bearer ' + accessToken } },
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
`

// Play a track via Spotify Connect Web API, then poll until it finishes.
const spotifyPlayJS = `
async ({ uri, durationSec }) => {
  const tokenRes = await fetch('/get_access_token');
  const { accessToken } = await tokenRes.json();
  const headers = { Authorization: 'Bearer ' + accessToken };

  // Find the web player device
  const devRes = await fetch(
    'https://api.spotify.com/v1/me/player/devices', { headers },
  );
  const { devices } = await devRes.json();
  const device = devices?.find(d => d.type === 'Computer');
  const qs = device ? '?device_id=' + device.id : '';

  // Start playback
  await fetch('https://api.spotify.com/v1/me/player/play' + qs, {
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
  return true;
}
`

const spotifyStopJS = `
async () => {
  const r = await fetch('/get_access_token');
  const { accessToken } = await r.json();
  await fetch('https://api.spotify.com/v1/me/player/pause', {
    method: 'PUT',
    headers: { Authorization: 'Bearer ' + accessToken },
  });
  return true;
}
`

// SpotifyMusic streams Spotify via open.spotify.com.
type SpotifyMusic struct {
	*BrowserSession
}

// NewSpotifyMusic creates an unstarted Spotify provider.
func NewSpotifyMusic(sampleRate int) *SpotifyMusic {
	return &SpotifyMusic{NewBrowserSession("https://open.spotify.com", "spotify-profile", sampleRate)}
}

type browserSearchResult struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	Artist   string `json:"artist"`
	Album    string `json:"album"`
	Duration int    `json:"duration"`
}

// Search implements BrowserMusic.
func (s *SpotifyMusic) Search(ctx context.Context, query string, limit int) ([]MusicTrack, error) {
	if limit <= 0 {
		limit = 20
	}
	var results []browserSearchResult
	if err := s.Evaluate(ctx, spotifySearchJS, &results, map[string]any{"query": query, "limit": limit}); err != nil {
		return nil, err
	}
	tracks := make([]MusicTrack, 0, len(results))
	for _, r := range results {
		tracks = append(tracks, MusicTrack{
			Title: r.Title, Artist: r.Artist, Album: r.Album,
			DurationSeconds: r.Duration, SongID: r.ID, Service: "spotify",
		})
	}
	return tracks, nil
}

// StreamTrack implements BrowserMusic.
func (s *SpotifyMusic) StreamTrack(ctx context.Context, track MusicTrack, output audio.PipelineOutput) error {
	return s.BrowserSession.StreamTrack(ctx, output, func(ctx context.Context) error {
		return s.Evaluate(ctx, spotifyPlayJS, nil, map[string]any{
			"uri": "spotify:track:" + track.SongID, "durationSec": track.DurationSeconds,
		})
	})
}

// StopPlayback implements BrowserMusic.
func (s *SpotifyMusic) StopPlayback(ctx context.Context) error {
	if s.pageCtx == nil {
		return nil
	}
	return s.Evaluate(ctx, spotifyStopJS, nil)
}
