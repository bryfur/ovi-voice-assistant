package browser

import (
	"context"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Provider is a music service driven through a Session by JS functions:
// search returns tracks, play resolves when a track ends, stop halts it.
type Provider struct {
	*Session
	name             string
	searchJS, playJS string
	playArg          func(music.Track) any
}

func (p *Provider) Search(ctx context.Context, query string, limit int) ([]music.Track, error) {
	return p.search(ctx, p.searchJS, p.name, query, limit)
}

func (p *Provider) Play(ctx context.Context, track music.Track, out device.Output) error {
	return p.play(ctx, out, p.playJS, p.playArg(track))
}
