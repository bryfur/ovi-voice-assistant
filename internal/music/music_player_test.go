package music

import (
	"context"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

type captureOutput struct {
	mu     sync.Mutex
	audio  [][]byte
	events []device.EventType
}

func (c *captureOutput) SendEvent(_ context.Context, e device.EventType, _ []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, e)
	return nil
}

func (c *captureOutput) SendAudio(_ context.Context, pcm []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.audio = append(c.audio, pcm)
	return nil
}

func (c *captureOutput) Flush(context.Context) error { return nil }
func (c *captureOutput) Reset()                      {}

func (c *captureOutput) totalBytes() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, a := range c.audio {
		n += len(a)
	}
	return n
}

func TestQueueOperations(t *testing.T) {
	p := NewMusicPlayer(48000, 2, nil)
	tracks := []MusicTrack{{Title: "A"}, {Title: "B"}}

	p.SetQueue(tracks, 0)

	if !p.IsActive() || p.QueueLen() != 2 || p.GetCurrent().Title != "A" {
		t.Fatal("SetQueue state wrong")
	}
	p.Pause()
	if p.IsActive() {
		t.Fatal("Pause should deactivate")
	}
	p.Resume()
	if !p.IsActive() {
		t.Fatal("Resume should activate")
	}
	if next := p.Skip(); next == nil || next.Title != "B" || p.CurrentIndex() != 1 {
		t.Fatal("Skip to B failed")
	}
	if end := p.Skip(); end != nil || p.IsActive() {
		t.Fatal("Skip past end should deactivate")
	}
	p.Stop()
	if p.QueueLen() != 0 || p.GetCurrent() != nil || p.CurrentIndex() != 0 {
		t.Fatal("Stop should clear")
	}
	p.Resume()
	if p.IsActive() {
		t.Fatal("Resume on empty queue should stay inactive")
	}
}

func TestStreamViaFakeFFmpeg(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script stub")
	}
	script := filepath.Join(t.TempDir(), "ffmpeg")
	os.WriteFile(script, []byte("#!/bin/sh\nhead -c 3840 /dev/zero\n"), 0o755)
	p := NewMusicPlayer(48000, 2, nil)
	p.FFmpegPath = script
	p.ExtractURL = func(context.Context, string) (string, error) { return "http://audio", nil }
	p.SetQueue([]MusicTrack{{Title: "A", VideoID: "x"}, {Title: "B", VideoID: "y"}}, 0)
	out := &captureOutput{}

	err := p.Stream(context.Background(), out)

	if err != nil || out.totalBytes() != 2*3840 || p.IsActive() || p.CurrentIndex() != 2 {
		t.Fatalf("err=%v bytes=%d active=%v idx=%d", err, out.totalBytes(), p.IsActive(), p.CurrentIndex())
	}
}

func TestStreamInactiveReturnsImmediately(t *testing.T) {
	p := NewMusicPlayer(48000, 2, nil)
	p.SetQueue([]MusicTrack{{Title: "A"}}, 0)
	p.Pause()

	err := p.Stream(context.Background(), &captureOutput{})

	if err != nil {
		t.Fatal(err)
	}
}

type fakeBrowser struct {
	streamed []MusicTrack
}

func (f *fakeBrowser) Search(context.Context, string, int) ([]MusicTrack, error) {
	return []MusicTrack{{Title: "B", Service: "fake"}}, nil
}

func (f *fakeBrowser) StreamTrack(ctx context.Context, track MusicTrack, out audioOutput) error {
	f.streamed = append(f.streamed, track)
	return out.SendAudio(ctx, []byte{1, 2})
}

func (f *fakeBrowser) StopPlayback(context.Context) error { return nil }

func TestStreamUsesBrowserProvider(t *testing.T) {
	fb := &fakeBrowser{}
	p := NewMusicPlayer(48000, 2, map[string]BrowserMusic{"fake": fb})
	p.SetQueue([]MusicTrack{{Title: "B", Service: "fake"}}, 0)
	out := &captureOutput{}

	err := p.Stream(context.Background(), out)

	if err != nil || len(fb.streamed) != 1 || out.totalBytes() != 2 {
		t.Fatalf("err=%v streamed=%d bytes=%d", err, len(fb.streamed), out.totalBytes())
	}
}
