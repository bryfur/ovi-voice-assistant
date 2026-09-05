package music

import (
	"context"
	"fmt"
	"sync"
)

// Browser-based music providers keyed by service name ("apple", "spotify").
var (
	browsersMu sync.RWMutex
	browsers   = map[string]BrowserMusic{}
)

// RegisterBrowser registers a browser music provider.
func RegisterBrowser(service string, b BrowserMusic) {
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
var SearchMusicFunc = SearchMusic

// ResolveYouTubeID finds the best YouTube Music video ID for a search query.
func ResolveYouTubeID(ctx context.Context, query string) (string, error) {
	tracks, err := SearchYouTube(ctx, query, 1)
	if err != nil {
		return "", err
	}
	if len(tracks) == 0 {
		return "", nil
	}
	return tracks[0].VideoID, nil
}

// SearchMusic searches for music; service selects the provider.
func SearchMusic(ctx context.Context, query, service string) ([]MusicTrack, error) {
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
		return SearchYouTube(ctx, query, 20)
	}
	available := []string{"youtube"}
	for k := range Browsers() {
		available = append(available, k)
	}
	return nil, fmt.Errorf("unknown music service %q (available: %v)", service, available)
}
