package music

import (
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
)

// syncLead is how far in the future devices are told to start playing, so
// the first frames have reached all of them.
const syncLead = 500 * time.Millisecond

// Player streams one queue of tracks to every registered device at once.
// Music never overlaps a voice session: the pipeline calls Interrupt when
// a wake word arrives and Continue once the session has finished talking,
// which is also when Play, Skip and Resume take effect.
type Player struct {
	services map[string]Service

	mu      sync.Mutex
	outputs []device.Output
	queue   []Track
	pos     int
	paused  bool
	cancel  context.CancelFunc // stops the streaming goroutine; nil when idle
	done    chan struct{}
}

// NewPlayer plays through YouTube Music plus the given services.
func NewPlayer(services map[string]Service) *Player {
	all := map[string]Service{"youtube": youtube{"yt-dlp", "ffmpeg"}}
	maps.Copy(all, services)
	return &Player{services: all}
}

// Services lists the available service names.
func (p *Player) Services() []string { return slices.Sorted(maps.Keys(p.services)) }

// AddOutput adds a device to play on.
func (p *Player) AddOutput(out device.Output) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.outputs = append(p.outputs, out)
}

// Search finds tracks on a service ("" means YouTube Music).
func (p *Player) Search(ctx context.Context, query, service string) ([]Track, error) {
	if service == "" {
		service = "youtube"
	}
	s, ok := p.services[service]
	if !ok {
		return nil, fmt.Errorf("unknown music service %q (available: %v)", service, p.Services())
	}
	return s.Search(ctx, query, 20)
}

// Play replaces the queue.
func (p *Player) Play(tracks []Track) {
	p.stop()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.queue, p.pos, p.paused = slices.Clone(tracks), 0, false
}

// Pause stops playback until Resume.
func (p *Player) Pause() {
	p.stop()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.paused = true
}

// Resume lets playback continue from the current track.
func (p *Player) Resume() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.paused = false
}

// Skip moves to the next track and returns it, or nil at the end of the queue.
func (p *Player) Skip() *Track {
	p.stop()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pos = min(p.pos+1, len(p.queue))
	return p.current()
}

// Stop ends playback and clears the queue.
func (p *Player) Stop() {
	p.stop()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.queue, p.pos = nil, 0
}

// Current is the track playing or paused, nil when the queue is done.
func (p *Player) Current() *Track {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.current()
}

// Remaining counts the tracks queued after the current one.
func (p *Player) Remaining() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return max(len(p.queue)-p.pos-1, 0)
}

// Paused reports whether the user paused playback.
func (p *Player) Paused() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.paused
}

// Interrupt stops streaming for a voice session; Continue picks the
// current track up again from the start.
func (p *Player) Interrupt() { p.stop() }

// Continue starts streaming if there is something to play.
func (p *Player) Continue() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cancel != nil || p.paused || p.current() == nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	p.cancel, p.done = cancel, done
	go func() {
		defer close(done)
		p.stream(ctx, fanout(slices.Clone(p.outputs)))
		p.mu.Lock()
		if p.done == done {
			p.cancel, p.done = nil, nil
		}
		p.mu.Unlock()
	}()
}

func (p *Player) current() *Track {
	if p.pos < len(p.queue) {
		return &p.queue[p.pos]
	}
	return nil
}

// stop cancels streaming, drops queued audio and tells the devices
// playback is over.
func (p *Player) stop() {
	p.mu.Lock()
	cancel, done, outs := p.cancel, p.done, fanout(slices.Clone(p.outputs))
	p.cancel, p.done = nil, nil
	p.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
	outs.Reset()
	_ = outs.SendEvent(tail(), device.EventTTSEnd, nil)
}

// stream plays from the current track to the end of the queue.
func (p *Player) stream(ctx context.Context, outs fanout) {
	outs.Reset()
	if err := outs.SendEvent(ctx, device.EventTTSStart, nil); err != nil {
		return
	}
	at := time.Now().Add(syncLead).UnixMilli()
	_ = outs.SendEvent(ctx, device.EventSyncPlay, binary.LittleEndian.AppendUint64(nil, uint64(at)))
	slog.Info("Music starting", "devices", len(outs), "at_ms", at)

	for t := p.Current(); t != nil && ctx.Err() == nil; t = p.Current() {
		slog.Info("Playing", "artist", t.Artist, "title", t.Title, "service", t.Service)
		s, ok := p.services[t.Service]
		if !ok {
			slog.Warn("Unknown music service", "service", t.Service)
		} else if err := s.Play(ctx, *t, outs); err != nil && ctx.Err() == nil {
			slog.Error("Track failed", "title", t.Title, "err", err)
		}
		if ctx.Err() != nil {
			return // interrupted: the track replays from the top on Continue
		}
		p.mu.Lock()
		p.pos++
		p.mu.Unlock()
	}
	_ = outs.Flush(ctx)
	_ = outs.SendEvent(tail(), device.EventTTSEnd, nil)
}

// tail is a short-lived context for the last words to a device.
func tail() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	context.AfterFunc(ctx, cancel)
	return ctx
}

// fanout sends the same audio and events to several devices.
type fanout []device.Output

func (f fanout) SendEvent(ctx context.Context, e device.Event, payload []byte) error {
	return f.each(func(o device.Output) error { return o.SendEvent(ctx, e, payload) })
}

func (f fanout) SendAudio(ctx context.Context, pcm []byte) error {
	return f.each(func(o device.Output) error { return o.SendAudio(ctx, pcm) })
}

func (f fanout) Flush(ctx context.Context) error {
	return f.each(func(o device.Output) error { return o.Flush(ctx) })
}

func (f fanout) Reset() {
	for _, o := range f {
		o.Reset()
	}
}

func (f fanout) each(fn func(device.Output) error) error {
	for _, o := range f {
		if err := fn(o); err != nil {
			return err
		}
	}
	return nil
}
