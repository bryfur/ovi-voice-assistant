package browser

import (
	"context"
	"log/slog"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Start launches and registers the configured browser providers
// ("spotify", "apple"). Apple Music login is awaited in the background.
// The returned func shuts them down.
func Start(ctx context.Context, names []string, sampleRate int) func() {
	var sessions []*Session
	for _, name := range names {
		var b music.BrowserMusic
		var session *Session
		switch name {
		case "spotify":
			s := NewSpotify(sampleRate)
			b, session = s, s.Session
		case "apple":
			a := NewApple(sampleRate)
			b, session = a, a.Session
			defer func() { go func() { _ = a.WaitForLogin(ctx) }() }()
		default:
			slog.Warn("Unknown music service", "name", name)
			continue
		}
		if err := session.Start(ctx); err != nil {
			slog.Error("Music service failed to start", "name", name, "err", err)
			continue
		}
		music.RegisterBrowser(name, b)
		sessions = append(sessions, session)
		slog.Info("Music service ready", "name", name)
	}
	return func() {
		for _, s := range sessions {
			s.Close()
		}
	}
}
