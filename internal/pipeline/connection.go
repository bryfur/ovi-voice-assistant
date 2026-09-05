package pipeline

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Voice is what a connection needs from the voice pipeline.
type Voice interface {
	Rate() int // sample rate of the speech produced
	Run(ctx context.Context, out device.Output, mic <-chan []byte, env *agent.Env) bool
	Announce(ctx context.Context, out device.Output, text string)
	Reset()
}

// micBuffer is how many mic frames may queue up for a session (~80 s).
const micBuffer = 8192

// Connection runs the voice loop for one device: it configures the
// device's audio, decodes its microphone and runs one session per wake
// word, pausing music while a session or announcement is playing.
type Connection struct {
	Name string
	// OnWake, when set, decides whether a wake word starts a session
	// (multi-device arbitration); otherwise every wake word does.
	OnWake func(c *Connection, score int, word string)
	// SetupDelay is the pause after connecting before the device is configured.
	SetupDelay time.Duration

	t       device.Transport
	voice   Voice
	speaker *device.Speaker // speech, at the voice format
	music   *device.Speaker // music, at 48 kHz stereo
	prefer  string          // preferred mic codec
	player  *music.Player
	env     *agent.Env

	mu         sync.Mutex
	mic        codec.Codec // set from the device's MIC_CONFIG
	queue      chan []byte // decoded mic PCM for the session
	cancel     context.CancelFunc
	done       chan struct{}
	continuing bool         // the last reply asked for a follow-up
	format     codec.Format // last speaker format announced, for logging
}

// NewConnection prepares a device that speaks the named codec; player may
// be nil when music is unavailable.
func NewConnection(t device.Transport, codecName string, v Voice, player *music.Player) (*Connection, error) {
	speech, err := codec.New(codecName, v.Rate(), 1, 0)
	if err != nil {
		return nil, err
	}
	tunes, err := codec.New(codecName, music.Rate, music.Channels, codec.MusicNByte)
	if err != nil {
		return nil, err
	}
	c := &Connection{
		Name: t.String(), SetupDelay: 500 * time.Millisecond,
		t: t, voice: v, prefer: codecName, player: player, queue: make(chan []byte, micBuffer),
	}
	c.speaker = device.NewSpeaker(t, speech)
	c.music = device.NewSpeaker(t, tunes)
	c.speaker.OnConfig, c.music.OnConfig = c.logFormat, c.logFormat
	c.env = &agent.Env{Announce: c.Announce, Music: player}
	if player != nil {
		player.AddOutput(c.music)
	}
	return c, nil
}

// Env is what the agent's tools see on this device.
func (c *Connection) Env() *agent.Env { return c.env }

// SetScheduler makes the automation tools available.
func (c *Connection) SetScheduler(s *scheduler.Scheduler) { c.env.Scheduler = s }

// Start connects and configures the device.
func (c *Connection) Start() error {
	err := c.t.Connect(device.Handler{Event: c.event, Audio: c.audio, Connect: c.configure, Disconnect: c.halt})
	if err != nil {
		return err
	}
	c.configure()
	return nil
}

// Stop ends any session and disconnects.
func (c *Connection) Stop() error {
	c.stop()
	return c.t.Disconnect()
}

// Announce speaks text on the device, interrupting a running session.
func (c *Connection) Announce(text string) {
	c.start(func(ctx context.Context) {
		c.interruptMusic()
		c.speaker.Reset()
		c.voice.Announce(ctx, c.speaker, text)
		_ = c.speaker.Flush(ctx)
		c.continueMusic(ctx)
	})
}

// StartSession begins a fresh conversation; the manager calls it for the
// device that won arbitration.
func (c *Connection) StartSession(word string) {
	c.voice.Reset()
	slog.Info("Voice session started", "device", c.Name, "wake_word", word)
	c.start(c.session)
}

// AbortWake stands a device down after it lost arbitration.
func (c *Connection) AbortWake() {
	c.stop()
	c.mu.Lock()
	c.queue = make(chan []byte, micBuffer) // drop what it heard
	c.mu.Unlock()
	_ = c.t.SendEvent(device.EventWakeAbort, nil)
	slog.Info("Wake aborted, another device won", "device", c.Name)
}

// Busy reports whether a session or announcement is running.
func (c *Connection) Busy() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	select {
	case <-c.done:
		return false
	default:
		return c.done != nil
	}
}

// configure tells the device the speaker format; it answers with its mic
// format (see micConfig).
func (c *Connection) configure() {
	time.Sleep(c.SetupDelay)
	f := c.speaker.Format()
	c.logFormat(f)
	if err := c.t.SendEvent(device.EventAudioConfig, device.AudioConfig(f)); err != nil {
		slog.Error("Device setup failed", "device", c.Name, "err", err)
	}
}

func (c *Connection) event(e device.Event, payload []byte) {
	switch e {
	case device.EventWakeWord:
		c.wake(payload)
	case device.EventMicConfig:
		c.micConfig(payload)
	}
}

// wake starts a session, or asks OnWake to arbitrate. A follow-up wake
// (the previous reply ended with a question) is never arbitrated.
func (c *Connection) wake(payload []byte) {
	c.stop()
	c.interruptMusic()
	c.mu.Lock()
	followUp := c.continuing
	c.continuing = false
	c.queue = make(chan []byte, micBuffer)
	c.mu.Unlock()
	word, score := device.ParseWake(payload)
	switch {
	case followUp:
		slog.Info("Follow-up session started", "device", c.Name)
		c.start(c.session)
	case c.OnWake != nil:
		c.OnWake(c, score, word)
	default:
		c.StartSession(word)
	}
}

// micConfig adopts the device's mic codec, or asks for the preferred one.
func (c *Connection) micConfig(payload []byte) {
	f, err := device.ParseFormat(payload)
	if err != nil {
		return
	}
	mic, err := codec.New(string(f.Type), f.Rate, 1, f.FrameBytes)
	if err != nil {
		slog.Error("Unusable mic codec", "device", c.Name, "err", err)
		return
	}
	slog.Info("Mic audio config (device)", "device", c.Name, "format", mic.Format())
	if string(f.Type) != c.prefer {
		if pref, err := codec.New(c.prefer, f.Rate, 1, 0); err == nil &&
			c.t.SendEvent(device.EventMicConfig, device.MicConfig(pref.Format())) == nil {
			mic = pref
			slog.Info("Mic audio config (requested)", "device", c.Name, "format", pref.Format())
		}
	}
	c.mu.Lock()
	c.mic = mic
	c.mu.Unlock()
}

// audio decodes a mic frame and queues it for the session.
func (c *Connection) audio(frame []byte) {
	c.mu.Lock()
	mic, queue := c.mic, c.queue
	c.mu.Unlock()
	pcm := frame
	if mic != nil {
		var err error
		if pcm, err = mic.Decode(frame); err != nil {
			slog.Debug("Mic decode failed", "err", err)
			return
		}
	}
	select {
	case queue <- pcm:
	default:
		slog.Debug("Mic queue full, dropping frame", "device", c.Name)
	}
}

// session runs one utterance, then lets music carry on unless the reply
// asked for a follow-up.
func (c *Connection) session(ctx context.Context) {
	c.mu.Lock()
	mic := c.queue
	c.mu.Unlock()
	c.speaker.Reset()
	again := c.voice.Run(ctx, c.speaker, mic, c.env)
	_ = c.speaker.Flush(ctx)
	c.mu.Lock()
	c.continuing = again
	c.mu.Unlock()
	if !again {
		c.continueMusic(ctx)
	}
}

func (c *Connection) interruptMusic() {
	if c.player != nil {
		c.player.Interrupt()
	}
}

func (c *Connection) continueMusic(ctx context.Context) {
	if c.player != nil && ctx.Err() == nil {
		c.player.Continue()
	}
}

// start runs task in the background after stopping the current one.
func (c *Connection) start(task func(context.Context)) {
	c.stop()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	c.mu.Lock()
	c.cancel, c.done = cancel, done
	c.mu.Unlock()
	go func() {
		defer close(done)
		task(ctx)
	}()
}

// stop cancels the running task and waits for it.
func (c *Connection) stop() {
	c.mu.Lock()
	cancel, done := c.cancel, c.done
	c.cancel, c.done = nil, nil
	c.mu.Unlock()
	if cancel != nil {
		cancel()
		c.speaker.Reset() // unblocks a task waiting on paced audio
		<-done
	}
}

// halt cancels the running task without waiting; used when the device
// drops off, from the transport's own goroutine.
func (c *Connection) halt() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel != nil {
		c.cancel()
	}
}

// logFormat reports the speaker format whenever it changes.
func (c *Connection) logFormat(f codec.Format) {
	c.mu.Lock()
	changed := f != c.format
	c.format = f
	c.mu.Unlock()
	if changed {
		slog.Info("Speaker audio config", "device", c.Name, "format", f)
	}
}
