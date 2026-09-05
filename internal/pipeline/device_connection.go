package pipeline

import (
	"context"
	"encoding/binary"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"log/slog"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// WakeCallback is invoked for wake-word arbitration: (connection, score, wakeWord).
type WakeCallback func(conn *DeviceConnection, score int, wakeWord string)

// micQueueSize bounds buffered mic chunks (~80 s of 10 ms frames).
const micQueueSize = 8192

// Runner is the voice pipeline surface a connection needs.
type Runner interface {
	Run(ctx context.Context, output device.Output, mic <-chan []byte, actx *agent.Context) bool
	Announce(ctx context.Context, output device.Output, text string)
	ResetHistory()
}

// DeviceConnection connects to a single voice device and routes audio
// through the pipeline.
type DeviceConnection struct {
	Name string

	transport device.Transport
	spkCodec  codec.AudioCodec
	pipeline  Runner
	settings  *config.Settings
	output    *device.EncodingOutput
	onWake    WakeCallback

	mu          sync.Mutex
	micCodec    codec.AudioCodec // created when device sends MIC_CONFIG
	lastSpeaker string           // last speaker config announced to the device
	audioQueue  chan []byte
	taskCancel  context.CancelFunc
	taskDone    chan struct{}
	continuing  bool
	context     *agent.Context
	musicOutput *device.EncodingOutput
	musicGroup  *music.MusicGroup

	// SetupDelay is the pause after (re)connect before device setup.
	SetupDelay time.Duration
}

// DeviceOptions configure a DeviceConnection.
type DeviceOptions struct {
	OnWake     WakeCallback
	Name       string
	MusicGroup *music.MusicGroup
}

// NewDeviceConnection creates an unstarted connection.
func NewDeviceConnection(t device.Transport, spkCodec codec.AudioCodec, p Runner, settings *config.Settings, opts DeviceOptions) *DeviceConnection {
	name := opts.Name
	if name == "" {
		name = t.String()
	}
	c := &DeviceConnection{
		Name:       name,
		transport:  t,
		spkCodec:   spkCodec,
		pipeline:   p,
		settings:   settings,
		output:     device.NewEncodingOutput(t, spkCodec),
		onWake:     opts.OnWake,
		audioQueue: make(chan []byte, micQueueSize),
		musicGroup: opts.MusicGroup,
		SetupDelay: 500 * time.Millisecond,
	}
	c.output.OnAudioConfig = c.speakerConfig
	return c
}

// speakerConfig logs the speaker settings whenever they change (voice ↔ music).
func (c *DeviceConnection) speakerConfig(cd codec.AudioCodec) {
	desc := codec.Describe(cd)
	c.mu.Lock()
	changed := desc != c.lastSpeaker
	c.lastSpeaker = desc
	c.mu.Unlock()
	if changed {
		slog.Info("Speaker audio config", "device", c.Name, "codec", desc)
	}
}

// Context returns the agent context for this device (nil before setup).
func (c *DeviceConnection) Context() *agent.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.context
}

// SetScheduler attaches the scheduler so automation tools are available.
func (c *DeviceConnection) SetScheduler(s *scheduler.Scheduler) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.context != nil {
		c.context.Scheduler = s
	}
}

// Start connects the transport and configures the device.
func (c *DeviceConnection) Start() error {
	c.transport.SetEventCallback(c.onEvent)
	c.transport.SetAudioCallback(c.onAudio)
	c.transport.SetDisconnectCallback(c.onDisconnect)
	c.transport.SetConnectCallback(c.onReconnect)
	if err := c.transport.Connect(); err != nil {
		return err
	}
	time.Sleep(c.SetupDelay)
	return c.setupDevice()
}

// Stop cancels any running task and disconnects.
func (c *DeviceConnection) Stop() error {
	c.cancelTask()
	return c.transport.Disconnect()
}

// setupDevice sends the speaker config and prepares the music path and
// agent context.
func (c *DeviceConnection) setupDevice() error {
	c.speakerConfig(c.spkCodec)
	if err := device.SendAudioConfig(c.transport, device.AudioConfig{
		SampleRate:        uint32(c.spkCodec.SampleRate()),
		EncodedFrameBytes: uint16(c.spkCodec.EncodedFrameBytes()),
		CodecType:         c.spkCodec.ID(),
		Channels:          uint8(c.spkCodec.Channels()),
	}); err != nil {
		return err
	}
	// Device responds with MIC_CONFIG after receiving AUDIO_CONFIG, which
	// configures micCodec via onEvent.

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.context != nil {
		return nil // already set up (reconnect)
	}
	// Music gets its own 48 kHz codec so playback is CD-quality+ while TTS
	// stays at whatever rate the voice codec uses. AUDIO_CONFIG is re-sent
	// before each TTS_START, so the device reconfigures its decoder.
	musicCodec, err := codec.NewCodec(c.settings.Transport.Codec, 48000, 2, codec.LC3MusicNByte)
	if err != nil {
		return fmt.Errorf("music codec: %w", err)
	}
	c.musicOutput = device.NewEncodingOutput(c.transport, musicCodec)
	c.musicOutput.OnAudioConfig = c.speakerConfig
	slog.Info("Music audio config", "device", c.Name, "codec", codec.Describe(musicCodec))
	if c.musicGroup != nil {
		c.musicGroup.AddDevice(c.musicOutput, c.transport)
	}
	player := music.NewMusicPlayer(musicCodec.SampleRate(), musicCodec.Channels(), music.Browsers())
	c.context = &agent.Context{
		Announce:    c.Announce,
		MusicPlayer: player,
		MusicGroup:  c.musicGroup,
	}
	return nil
}

// -- Transport callbacks --

func (c *DeviceConnection) onEvent(event device.EventType, payload []byte) {
	switch event {
	case device.EventWakeWord:
		c.handleWake(payload)
	case device.EventMicConfig:
		c.handleMicConfig(payload)
	}
}

func (c *DeviceConnection) handleWake(payload []byte) {
	c.cancelTask()

	// Pause group music on ALL devices when any device wakes
	if c.musicGroup != nil && c.musicGroup.IsActive() {
		c.musicGroup.Pause(context.Background())
	}

	c.mu.Lock()
	continuing := c.continuing
	c.continuing = false
	c.audioQueue = make(chan []byte, micQueueSize)
	c.mu.Unlock()

	if continuing {
		// Follow-up — already committed to this device, skip arbitration.
		slog.Info("Follow-up session started")
		c.startTask(c.run)
		return
	}

	// Parse extended payload: [2B LE peak][2B LE ambient][wake_word UTF-8]
	// The normalised score (peak/ambient) cancels out mic gain differences
	// so the *closest* device wins, not the loudest mic.
	score := 0
	wakeWord := ""
	if len(payload) >= 4 {
		peak := int(binary.LittleEndian.Uint16(payload[0:]))
		ambient := int(binary.LittleEndian.Uint16(payload[2:]))
		wakeWord = string(payload[4:])
		if ambient < 1 {
			ambient = 1
		}
		score = peak * 1000 / ambient
		slog.Debug("Wake payload", "peak", peak, "ambient", ambient, "score", score)
	} else if len(payload) > 0 {
		wakeWord = string(payload)
	}

	if c.onWake != nil {
		c.onWake(c, score, wakeWord)
		return
	}
	// No arbitration — start immediately.
	c.pipeline.ResetHistory()
	slog.Info("Voice session started", "wake_word", wakeWord)
	c.startTask(c.run)
}

func (c *DeviceConnection) handleMicConfig(payload []byte) {
	if len(payload) < 7 {
		return
	}
	rate := int(binary.LittleEndian.Uint32(payload[0:]))
	nbyte := int(binary.LittleEndian.Uint16(payload[4:]))
	name := codec.NameForID(payload[6])

	// Use the device's reported mic codec
	mic, err := codec.NewCodec(string(name), rate, 1, nbyte)
	if err != nil {
		slog.Error("Cannot create mic codec", "codec", name, "err", err)
		return
	}
	slog.Info("Mic audio config (device)", "device", c.Name, "codec", codec.Describe(mic))

	// If the server prefers a different mic codec, request it
	preferred := c.settings.Transport.Codec
	if preferred != string(name) {
		pref, err := codec.NewCodec(preferred, rate, 1, 0)
		if err == nil {
			buf := make([]byte, 7)
			binary.LittleEndian.PutUint32(buf[0:], uint32(rate))
			binary.LittleEndian.PutUint16(buf[4:], uint16(pref.EncodedFrameBytes()))
			buf[6] = pref.ID()
			if err := c.transport.SendEvent(device.EventMicConfig, buf); err == nil {
				mic = pref
				slog.Info("Mic audio config (requested)", "device", c.Name, "codec", codec.Describe(pref))
			}
		}
	}
	c.mu.Lock()
	c.micCodec = mic
	c.mu.Unlock()
}

// onAudio decodes mic audio and queues PCM for the pipeline.
func (c *DeviceConnection) onAudio(data []byte) {
	c.mu.Lock()
	mic := c.micCodec
	queue := c.audioQueue
	c.mu.Unlock()
	pcm := data
	if mic != nil {
		decoded, err := mic.Decode(data)
		if err != nil {
			slog.Debug("Mic decode failed", "err", err)
			return
		}
		pcm = decoded
	}
	select {
	case queue <- pcm:
	default:
		slog.Debug("Mic queue full, dropping frame", "device", c.Name)
	}
}

func (c *DeviceConnection) onDisconnect() {
	c.mu.Lock()
	cancel := c.taskCancel
	c.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (c *DeviceConnection) onReconnect() {
	time.Sleep(c.SetupDelay)
	if err := c.setupDevice(); err != nil {
		slog.Error("Device setup after reconnect failed", "device", c.Name, "err", err)
	}
}

// -- Pipeline execution --

// run executes one voice session on this device.
func (c *DeviceConnection) run(ctx context.Context) {
	c.mu.Lock()
	mic := c.audioQueue
	actx := c.context
	c.mu.Unlock()

	c.output.Reset()
	continuing := c.pipeline.Run(ctx, c.output, mic, actx)
	_ = c.output.Flush(ctx)

	c.mu.Lock()
	c.continuing = continuing
	c.mu.Unlock()

	// After the voice pipeline completes, resume music if it was playing
	if !continuing && actx != nil && ctx.Err() == nil {
		if c.musicGroup != nil && c.musicGroup.IsActive() {
			// Group music was paused for voice — resume on all devices
			c.musicGroup.Resume(ctx)
		} else if actx.MusicPlayer != nil && actx.MusicPlayer.IsActive() {
			c.runMusic(ctx, actx.MusicPlayer)
		}
	}
}

// runMusic streams music at 48 kHz via the dedicated music output.
func (c *DeviceConnection) runMusic(ctx context.Context, player *music.MusicPlayer) {
	out := c.musicOutput
	out.Reset()
	if err := out.SendEvent(ctx, device.EventTTSStart, nil); err != nil {
		return
	}
	err := player.Stream(ctx, out)
	// Always try to close the speaker, even after cancellation.
	tail, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if ferr := out.Flush(tail); ferr == nil {
		_ = out.SendEvent(tail, device.EventTTSEnd, nil)
	}
	if err != nil && ctx.Err() == nil {
		slog.Error("Music streaming error", "device", c.Name, "err", err)
	}
}

func (c *DeviceConnection) runAnnounce(text string) func(ctx context.Context) {
	return func(ctx context.Context) {
		c.output.Reset()
		c.pipeline.Announce(ctx, c.output, text)
		_ = c.output.Flush(ctx)
	}
}

// -- Helpers --

// startTask cancels any running task and starts fn in a new goroutine.
func (c *DeviceConnection) startTask(fn func(ctx context.Context)) {
	c.cancelTask()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	c.mu.Lock()
	c.taskCancel = cancel
	c.taskDone = done
	c.mu.Unlock()
	go func() {
		defer close(done)
		fn(ctx)
	}()
}

// cancelTask cancels the running task and waits for it to finish.
func (c *DeviceConnection) cancelTask() {
	c.mu.Lock()
	cancel := c.taskCancel
	done := c.taskDone
	c.taskCancel = nil
	c.taskDone = nil
	c.mu.Unlock()
	if cancel != nil {
		cancel()
		// The task may be blocked on the paced output; reset it so it
		// unwinds promptly, then wait.
		c.output.Reset()
		<-done
	}
}

// TaskRunning reports whether a session/announcement is in progress.
func (c *DeviceConnection) TaskRunning() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.taskDone == nil {
		return false
	}
	select {
	case <-c.taskDone:
		return false
	default:
		return true
	}
}

// StartPipeline is called by DeviceManager when this device wins arbitration.
func (c *DeviceConnection) StartPipeline(wakeWord string) {
	c.pipeline.ResetHistory()
	slog.Info("Voice session started (won arbitration)", "device", c.Name, "wake_word", wakeWord)
	c.startTask(c.run)
}

// AbortWake is called by DeviceManager when this device loses arbitration.
func (c *DeviceConnection) AbortWake() {
	c.cancelTask()
	// Drain any buffered mic audio
	c.mu.Lock()
	queue := c.audioQueue
	c.mu.Unlock()
	for {
		select {
		case <-queue:
			continue
		default:
		}
		break
	}
	_ = c.transport.SendEvent(device.EventWakeAbort, nil)
	slog.Info("Wake aborted (another device won)", "device", c.Name)
}

// Announce streams a TTS announcement directly to the device, interrupting
// any running session.
func (c *DeviceConnection) Announce(text string) {
	c.startTask(c.runAnnounce(text))
}

// AnnounceAndWait announces and blocks until playback finishes or ctx ends.
func (c *DeviceConnection) AnnounceAndWait(ctx context.Context, text string) error {
	c.Announce(text)
	c.mu.Lock()
	done := c.taskDone
	c.mu.Unlock()
	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
