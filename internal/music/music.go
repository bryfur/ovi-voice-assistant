// Package music finds and plays music on the connected devices: YouTube
// Music through yt-dlp and ffmpeg, Spotify and Apple Music through a
// captured browser tab (see the browser subpackage).
package music

import (
	"context"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
)

// Music plays at a little better than CD quality, in stereo.
const (
	Rate     = 48000
	Channels = 2
)

// Track is one playable song.
type Track struct {
	Title, Artist, Album string
	Duration             int    // seconds
	ID                   string // YouTube video id, or the service's own song id
	Service              string // "youtube", "spotify" or "apple"
}

// Service finds tracks and plays them.
type Service interface {
	Search(ctx context.Context, query string, limit int) ([]Track, error)
	// Play streams track into out until it ends or ctx is cancelled.
	Play(ctx context.Context, track Track, out device.Output) error
}
