package browser

import (
	"context"
	"log/slog"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Start launches the named services ("spotify", "apple") and returns the
// ones that came up, plus a func that closes them.
func Start(ctx context.Context, names []string) (map[string]music.Service, func()) {
	services := map[string]music.Service{}
	var sessions []*Session
	for _, name := range names {
		var p *Provider
		switch name {
		case "spotify":
			p = Spotify()
		case "apple":
			p = Apple()
		default:
			slog.Warn("Unknown music service", "name", name)
			continue
		}
		if err := p.launch(); err != nil {
			slog.Error("Music service failed to start", "name", name, "err", err)
			continue
		}
		if name == "apple" {
			go p.waitForLogin(ctx)
		}
		services[name], sessions = p, append(sessions, p.Session)
		slog.Info("Music service ready", "name", name)
	}
	return services, func() {
		for _, s := range sessions {
			s.Close()
		}
	}
}
