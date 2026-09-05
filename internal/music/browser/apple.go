package browser

import (
	"context"
	"log/slog"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Apple plays through music.apple.com using the page's MusicKit instance.
func Apple() *Provider {
	return &Provider{
		Session:  newSession("https://music.apple.com", "apple-music-profile", appleStopJS),
		name:     "apple",
		searchJS: appleSearchJS,
		playJS:   applePlayJS,
		playArg:  func(t music.Track) any { return t.ID },
	}
}

const appleSearchJS = `
async ({ query, limit }) => {
  const mk = window.MusicKit?.getInstance?.();
  if (!mk) throw new Error('MusicKit not available — log in first');
  const res = await mk.api.music('/v1/catalog/' + mk.storefrontId + '/search',
    { term: query, types: 'songs', limit });
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
    if (state === MusicKit.PlaybackStates.completed || state === MusicKit.PlaybackStates.ended) {
      mk.removeEventListener('playbackStateDidChange', done);
      resolve(true);
    }
  };
  mk.addEventListener('playbackStateDidChange', done);
  mk.setQueue({ song: songId }).then(() => mk.play());
})
`

const appleStopJS = `() => { MusicKit.getInstance()?.stop(); return true; }`

// waitForLogin logs when MusicKit reports an authorized user, prompting
// for a sign-in in the browser window until then.
func (p *Provider) waitForLogin(ctx context.Context) {
	authorized := func() bool {
		var ok bool
		_ = p.eval(ctx, "() => !!window.MusicKit?.getInstance?.()?.isAuthorized", &ok)
		return ok
	}
	if authorized() {
		slog.Info("Apple Music: already logged in")
		return
	}
	slog.Info("Apple Music: waiting for login — sign in via the browser window")
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
		if authorized() {
			slog.Info("Apple Music: login successful")
			return
		}
	}
}
