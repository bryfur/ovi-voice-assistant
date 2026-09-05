package music

import (
	"context"
	"testing"
)

func TestRegisterBrowserAndSearchDispatch(t *testing.T) {
	fb := &fakeBrowser{}
	RegisterBrowser("fake", fb)
	defer func() {
		browsersMu.Lock()
		delete(browsers, "fake")
		browsersMu.Unlock()
	}()

	tracks, err := SearchMusic(context.Background(), "q", "fake")

	if err != nil || len(tracks) != 1 || tracks[0].Service != "fake" {
		t.Fatalf("got %+v, %v", tracks, err)
	}
	if _, ok := Browsers()["fake"]; !ok {
		t.Fatal("Browsers() should include registered provider")
	}
}

func TestSearchMusicUnknownService(t *testing.T) {
	_, err := SearchMusic(context.Background(), "q", "tidal")

	if err == nil {
		t.Fatal("expected error")
	}
}
