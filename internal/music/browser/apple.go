package browser

import (
	"context"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"log/slog"
	"time"
)

const appleSearchJS = `
async ({ query, limit }) => {
  const mk = window.MusicKit?.getInstance?.();
  if (!mk) throw new Error('MusicKit not available — log in first');
  const res = await mk.api.music(
    '/v1/catalog/' + mk.storefrontId + '/search',
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
`

const applePlayJS = `
(songId) => new Promise(resolve => {
  const mk = MusicKit.getInstance();
  const done = ({ state }) => {
    if (state === MusicKit.PlaybackStates.completed
        || state === MusicKit.PlaybackStates.ended) {
      mk.removeEventListener('playbackStateDidChange', done);
      resolve(true);
    }
  };
  mk.addEventListener('playbackStateDidChange', done);
  mk.setQueue({ song: songId }).then(() => mk.play());
})
`

// Apple streams Apple Music via music.apple.com.
type Apple struct {
	*Session
}

// NewApple creates an unstarted Apple Music provider.
func NewApple(sampleRate int) *Apple {
	return &Apple{NewSession("https://music.apple.com", "apple-music-profile", sampleRate)}
}

// Search implements BrowserMusic.
func (a *Apple) Search(ctx context.Context, query string, limit int) ([]music.MusicTrack, error) {
	if limit <= 0 {
		limit = 20
	}
	var results []browserSearchResult
	if err := a.Evaluate(ctx, appleSearchJS, &results, map[string]any{"query": query, "limit": limit}); err != nil {
		return nil, err
	}
	tracks := make([]music.MusicTrack, 0, len(results))
	for _, r := range results {
		tracks = append(tracks, music.MusicTrack{
			Title: r.Title, Artist: r.Artist, Album: r.Album,
			DurationSeconds: r.Duration, SongID: r.ID, Service: "apple",
		})
	}
	return tracks, nil
}

// StreamTrack implements BrowserMusic.
func (a *Apple) StreamTrack(ctx context.Context, track music.MusicTrack, output device.Output) error {
	return a.Session.StreamTrack(ctx, output, func(ctx context.Context) error {
		return a.Evaluate(ctx, applePlayJS, nil, track.SongID)
	})
}

// StopPlayback implements BrowserMusic.
func (a *Apple) StopPlayback(ctx context.Context) error {
	if a.pageCtx == nil {
		return nil
	}
	return a.Evaluate(ctx, "() => { MusicKit.getInstance()?.stop(); return true; }", nil)
}

// WaitForLogin polls until MusicKit reports an authorized user.
func (a *Apple) WaitForLogin(ctx context.Context) error {
	ready := false
	for i := 0; i < 10; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
		if err := a.Evaluate(ctx, "() => typeof MusicKit !== 'undefined' && !!MusicKit.getInstance?.()", &ready); err == nil && ready {
			break
		}
	}
	if !ready {
		slog.Warn("MusicKit not found on page")
		return nil
	}
	authorized := false
	if err := a.Evaluate(ctx, "() => MusicKit.getInstance().isAuthorized", &authorized); err == nil && authorized {
		slog.Info("Apple Music: already logged in")
		return nil
	}
	slog.Info("Apple Music: waiting for login — sign in via the browser window")
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
		if err := a.Evaluate(ctx, "() => MusicKit.getInstance().isAuthorized", &authorized); err == nil && authorized {
			slog.Info("Apple Music: login successful")
			return nil
		}
	}
}
