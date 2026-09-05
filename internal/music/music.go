package music

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
)

// Browser-based music providers keyed by service name ("apple", "spotify").
var (
	browsersMu sync.RWMutex
	browsers   = map[string]BrowserMusic{}
)

// registerBrowser registers a browser music provider.
func registerBrowser(service string, b BrowserMusic) {
	browsersMu.Lock()
	defer browsersMu.Unlock()
	browsers[service] = b
}

// Browsers returns registered browser providers (for passing to
// MusicPlayer/MusicGroup).
func Browsers() map[string]BrowserMusic {
	browsersMu.RLock()
	defer browsersMu.RUnlock()
	out := make(map[string]BrowserMusic, len(browsers))
	for k, v := range browsers {
		out[k] = v
	}
	return out
}

// SearchMusicFunc is the search implementation used by the play_music tool.
// Tests may replace it.
var SearchMusicFunc = searchMusic

// StartServices launches and registers the configured browser providers
// ("spotify", "apple"). Apple Music login is awaited in the background.
// The returned func shuts them down.
func StartServices(ctx context.Context, names []string, sampleRate int) func() {
	var sessions []*browserSession
	for _, name := range names {
		var b BrowserMusic
		var session *browserSession
		switch name {
		case "spotify":
			s := newSpotifyMusic(sampleRate)
			b, session = s, s.browserSession
		case "apple":
			a := newAppleMusic(sampleRate)
			b, session = a, a.browserSession
			defer func() { go func() { _ = a.WaitForLogin(ctx) }() }()
		default:
			slog.Warn("Unknown music service", "name", name)
			continue
		}
		if err := session.Start(ctx); err != nil {
			slog.Error("Music service failed to start", "name", name, "err", err)
			continue
		}
		registerBrowser(name, b)
		sessions = append(sessions, session)
		slog.Info("Music service ready", "name", name)
	}
	return func() {
		for _, s := range sessions {
			s.Close()
		}
	}
}

// searchMusic searches for music; service selects the provider.
func searchMusic(ctx context.Context, query, service string) ([]MusicTrack, error) {
	if service == "" {
		service = "youtube"
	}
	browsersMu.RLock()
	b, ok := browsers[service]
	browsersMu.RUnlock()
	if ok {
		return b.Search(ctx, query, 20)
	}
	if service == "youtube" {
		return searchYouTube(ctx, query, 20)
	}
	available := []string{"youtube"}
	for k := range Browsers() {
		available = append(available, k)
	}
	return nil, fmt.Errorf("unknown music service %q (available: %v)", service, available)
}
