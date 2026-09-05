package music

import (
	"context"
	"encoding/binary"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"log/slog"
	"sync"
	"time"
)

// syncBufferMs is how far in the future to schedule playback start. It must
// be large enough for audio to reach all devices and buffer.
const syncBufferMs = 500

// fanoutOutput sends the same PCM and events to multiple device outputs.
type fanoutOutput struct {
	outputs []device.Speaker
}

func (f *fanoutOutput) SendEvent(ctx context.Context, event device.EventType, payload []byte) error {
	var wg sync.WaitGroup
	errs := make([]error, len(f.outputs))
	for i, o := range f.outputs {
		wg.Add(1)
		go func(i int, o device.Speaker) {
			defer wg.Done()
			errs[i] = o.SendEvent(ctx, event, payload)
		}(i, o)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

func (f *fanoutOutput) SendAudio(ctx context.Context, pcm []byte) error {
	var wg sync.WaitGroup
	errs := make([]error, len(f.outputs))
	for i, o := range f.outputs {
		wg.Add(1)
		go func(i int, o device.Speaker) {
			defer wg.Done()
			errs[i] = o.SendAudio(ctx, pcm)
		}(i, o)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// MusicGroup coordinates music playback across multiple devices.
//
// All devices sync their clocks via NTP. The server sends audio to every
// device simultaneously, then sends a SYNC_PLAY event with a future NTP
// timestamp. Each device buffers audio and starts playback at the same
// wall-clock moment.
//
// It owns a single MusicPlayer (one ffmpeg process) and fans out PCM audio
// to all grouped device outputs simultaneously.
type MusicGroup struct {
	Player *MusicPlayer

	mu         sync.Mutex
	outputs    []device.Speaker
	transports []device.Transport
	cancel     context.CancelFunc
	done       chan struct{}
}

// NewMusicGroup creates a group producing PCM at sampleRate/channels.
func NewMusicGroup(sampleRate, channels int, browsers map[string]BrowserMusic) *MusicGroup {
	return &MusicGroup{Player: NewMusicPlayer(sampleRate, channels, browsers)}
}

// AddDevice registers a device's music output and transport.
func (g *MusicGroup) AddDevice(output device.Speaker, t device.Transport) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.outputs = append(g.outputs, output)
	g.transports = append(g.transports, t)
}

// DeviceCount returns the number of registered devices.
func (g *MusicGroup) DeviceCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.outputs)
}

// Play queues tracks for playback. Streaming starts after the voice
// pipeline finishes (via DeviceConnection), so TTS and music never compete
// for the same decoder.
func (g *MusicGroup) Play(ctx context.Context, tracks []MusicTrack) {
	g.Stop(ctx)
	g.Player.SetQueue(tracks, 0)
}

// IsActive reports whether the group is playing.
func (g *MusicGroup) IsActive() bool { return g.Player.IsActive() }

func (g *MusicGroup) cancelTask() {
	g.mu.Lock()
	cancel := g.cancel
	done := g.done
	g.cancel = nil
	g.done = nil
	g.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
}

func (g *MusicGroup) endAll(ctx context.Context) {
	g.mu.Lock()
	outputs := append([]device.Speaker(nil), g.outputs...)
	g.mu.Unlock()
	for _, out := range outputs {
		if err := out.Flush(ctx); err != nil {
			slog.Debug("music flush failed", "err", err)
			continue
		}
		if err := out.SendEvent(ctx, device.EventTTSEnd, nil); err != nil {
			slog.Debug("music TTS_END failed", "err", err)
		}
	}
}

// Pause pauses playback on all devices.
func (g *MusicGroup) Pause(ctx context.Context) {
	g.Player.Pause()
	g.cancelTask()
	// Signal all devices to stop speaker
	g.endAll(ctx)
}

// Resume restarts synchronized playback on all devices.
func (g *MusicGroup) Resume(ctx context.Context) {
	g.Player.Resume()
	if g.Player.IsActive() {
		g.startTask()
	}
}

// Skip advances to the next track and restarts streaming if active.
func (g *MusicGroup) Skip() *MusicTrack {
	track := g.Player.Skip()
	g.mu.Lock()
	running := g.cancel != nil
	g.mu.Unlock()
	if track != nil && running {
		g.cancelTask()
		g.startTask()
	}
	return track
}

// Stop halts playback and clears the queue on all devices.
func (g *MusicGroup) Stop(ctx context.Context) {
	g.Player.Stop()
	g.cancelTask()
	g.endAll(ctx)
}

// Close stops the group and releases resources.
func (g *MusicGroup) Close(ctx context.Context) {
	g.Stop(ctx)
}

func (g *MusicGroup) startTask() {
	g.mu.Lock()
	if g.cancel != nil {
		g.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	g.cancel = cancel
	g.done = done
	g.mu.Unlock()
	go func() {
		defer close(done)
		g.streamAll(ctx)
	}()
}

// streamAll streams music to all devices with synchronized start.
func (g *MusicGroup) streamAll(ctx context.Context) {
	g.mu.Lock()
	outputs := append([]device.Speaker(nil), g.outputs...)
	transports := append([]device.Transport(nil), g.transports...)
	g.mu.Unlock()
	fanout := &fanoutOutput{outputs: outputs}

	for _, out := range outputs {
		out.Reset()
	}

	// Send TTS_START (which also sends AUDIO_CONFIG) to all devices
	if err := fanout.SendEvent(ctx, device.EventTTSStart, nil); err != nil {
		return
	}

	// Send SYNC_PLAY with a future NTP timestamp to all devices
	targetMs := time.Now().UnixMilli() + syncBufferMs
	payload := make([]byte, 8)
	binary.LittleEndian.PutUint64(payload, uint64(targetMs))
	for _, t := range transports {
		if err := t.SendEvent(device.EventSyncPlay, payload); err != nil {
			slog.Error("Failed to send SYNC_PLAY", "err", err)
		}
	}
	slog.Info("Sync playback scheduled", "target_ms", targetMs, "buffer_ms", syncBufferMs, "devices", len(outputs))

	err := g.Player.Stream(ctx, fanout)
	// Use a fresh context for the tail: the stream context may be cancelled.
	tail, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, out := range outputs {
		if ferr := out.Flush(tail); ferr != nil {
			continue
		}
		_ = out.SendEvent(tail, device.EventTTSEnd, nil)
	}
	if err != nil && ctx.Err() == nil {
		slog.Error("Group streaming error", "err", err)
	}
}
