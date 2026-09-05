package music

import (
	"context"
	"encoding/binary"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
)

// sink records what a device would receive.
type sink struct {
	mu     sync.Mutex
	events []device.Event
	bytes  int
	sync   int64
}

func (s *sink) SendEvent(_ context.Context, e device.Event, payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, e)
	if e == device.EventSyncPlay {
		s.sync = int64(binary.LittleEndian.Uint64(payload))
	}
	return nil
}

func (s *sink) SendAudio(_ context.Context, pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bytes += len(pcm)
	return nil
}

func (s *sink) Flush(context.Context) error { return nil }
func (s *sink) Reset()                      {}

func (s *sink) got() ([]device.Event, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.events), s.bytes
}

// fakeService plays every track as two bytes, or blocks until cancelled.
type fakeService struct {
	block   bool
	started chan string // receives each title as it starts playing
	mu      sync.Mutex
	played  []string
}

func (f *fakeService) Search(context.Context, string, int) ([]Track, error) {
	return []Track{{Title: "Found", Service: "fake"}}, nil
}

func (f *fakeService) Play(ctx context.Context, t Track, out device.Output) error {
	f.mu.Lock()
	f.played = append(f.played, t.Title)
	f.mu.Unlock()
	if f.started != nil {
		f.started <- t.Title
	}
	if f.block {
		<-ctx.Done()
		return ctx.Err()
	}
	return out.SendAudio(ctx, []byte{1, 2})
}

func wait(t *testing.T, p *Player) {
	t.Helper()
	p.mu.Lock()
	done := p.done
	p.mu.Unlock()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stream did not finish")
	}
}

func TestQueueOperations(t *testing.T) {
	p := NewPlayer(nil)

	p.Play([]Track{{Title: "A"}, {Title: "B"}})

	if p.Current().Title != "A" || p.Remaining() != 1 || p.Paused() {
		t.Fatal("Play state wrong")
	}
	p.Pause()
	p.Resume()
	if next := p.Skip(); next == nil || next.Title != "B" || p.Remaining() != 0 {
		t.Fatal("Skip to B failed")
	}
	if p.Skip() != nil || p.Current() != nil {
		t.Fatal("Skip past the end should leave nothing current")
	}
	p.Stop()
	if p.Current() != nil || len(p.queue) != 0 {
		t.Fatal("Stop should clear")
	}
}

func TestContinueStreamsToAllDevicesInSync(t *testing.T) {
	svc := &fakeService{}
	p := NewPlayer(map[string]Service{"fake": svc})
	a, b := &sink{}, &sink{}
	p.AddOutput(a)
	p.AddOutput(b)
	p.Play([]Track{{Title: "A", Service: "fake"}, {Title: "B", Service: "fake"}})
	before := time.Now().Add(syncLead).UnixMilli()

	p.Continue()
	wait(t, p)

	want := []device.Event{device.EventTTSStart, device.EventSyncPlay, device.EventTTSEnd}
	for _, s := range []*sink{a, b} {
		events, n := s.got()
		if !slices.Equal(events, want) || n != 4 || s.sync < before {
			t.Fatalf("events=%v bytes=%d sync=%d", events, n, s.sync)
		}
	}
	if !slices.Equal(svc.played, []string{"A", "B"}) || p.Current() != nil {
		t.Fatalf("played %v, current %v", svc.played, p.Current())
	}
}

func TestInterruptKeepsTrackAndEndsPlayback(t *testing.T) {
	svc := &fakeService{block: true, started: make(chan string, 2)}
	p := NewPlayer(map[string]Service{"fake": svc})
	s := &sink{}
	p.AddOutput(s)
	p.Play([]Track{{Title: "A", Service: "fake"}})
	p.Continue()
	<-svc.started

	p.Interrupt()

	events, _ := s.got()
	if events[len(events)-1] != device.EventTTSEnd || p.Current().Title != "A" {
		t.Fatalf("events=%v current=%v", events, p.Current())
	}
	p.Continue()
	if again := <-svc.started; again != "A" {
		t.Fatalf("track should replay after interrupt, got %q", again)
	}
	p.Stop()
}

func TestContinueRespectsPauseAndEmptyQueue(t *testing.T) {
	p := NewPlayer(map[string]Service{"fake": &fakeService{block: true}})
	p.Continue() // nothing queued
	p.Play([]Track{{Title: "A", Service: "fake"}})
	p.Pause()

	p.Continue()

	if p.cancel != nil {
		t.Fatal("paused player must not stream")
	}
}

func TestSearchDispatch(t *testing.T) {
	p := NewPlayer(map[string]Service{"fake": &fakeService{}})

	tracks, err := p.Search(context.Background(), "q", "fake")

	if err != nil || len(tracks) != 1 || tracks[0].Service != "fake" {
		t.Fatalf("got %+v, %v", tracks, err)
	}
	if _, err := p.Search(context.Background(), "q", "tidal"); err == nil {
		t.Fatal("expected error for unknown service")
	}
	if !slices.Equal(p.Services(), []string{"fake", "youtube"}) {
		t.Fatalf("services = %v", p.Services())
	}
}
