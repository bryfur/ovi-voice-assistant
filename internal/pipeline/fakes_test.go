package pipeline

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
)

var ctx = context.Background()

// slowTTS emits each sentence's text as PCM after a small delay.
type slowTTS struct {
	delay time.Duration
	err   error
}

func (s *slowTTS) Load() error     { return nil }
func (s *slowTTS) SampleRate() int { return 16000 }
func (s *slowTTS) Close()          {}
func (s *slowTTS) Synthesize(text string, emit func([]byte) error) error {
	if s.err != nil {
		return s.err
	}
	time.Sleep(s.delay)
	return emit([]byte(text))
}

type fakeSTT struct {
	transcript string
	err        error
	speech     bool
	closed     bool
}

func (f *fakeSTT) Load() error { return nil }
func (f *fakeSTT) Close()      { f.closed = true }
func (f *fakeSTT) Listen(_ context.Context, _ <-chan []byte, onSpeech func()) (string, error) {
	if f.speech && onSpeech != nil {
		onSpeech()
	}
	return f.transcript, f.err
}

// fakeAgent streams its response word by word, pausing per word when slow.
type fakeAgent struct {
	response string
	err      error
	slow     time.Duration
	inputs   []string
	resets   int
}

func (a *fakeAgent) Load() error                 { return nil }
func (a *fakeAgent) Start(context.Context) error { return nil }
func (a *fakeAgent) Stop()                       {}
func (a *fakeAgent) Reset()                      { a.resets++ }
func (a *fakeAgent) Ask(context.Context, string, *agent.Env) (string, error) {
	return a.response, a.err
}
func (a *fakeAgent) Run(_ context.Context, text string, _ *agent.Env, emit func(string)) error {
	a.inputs = append(a.inputs, text)
	if a.err != nil {
		return a.err
	}
	for _, w := range strings.SplitAfter(a.response, " ") {
		emit(w)
		time.Sleep(a.slow)
	}
	return nil
}

// recordingOutput captures events and audio, with the time of each.
type recordingOutput struct {
	mu     sync.Mutex
	start  time.Time
	audio  []string
	events []device.Event
	times  []time.Duration
}

func (r *recordingOutput) SendEvent(_ context.Context, e device.Event, _ []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
	r.times = append(r.times, time.Since(r.start))
	return nil
}

func (r *recordingOutput) SendAudio(_ context.Context, pcm []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.audio = append(r.audio, string(pcm))
	r.times = append(r.times, time.Since(r.start))
	return nil
}

func (r *recordingOutput) Flush(context.Context) error { return nil }
func (r *recordingOutput) Reset()                      {}

func (r *recordingOutput) played() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.audio...)
}

func (r *recordingOutput) eventNames() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, len(r.events))
	for i, e := range r.events {
		names[i] = e.String()
	}
	return strings.Join(names, ",")
}

// fakeTransport records outbound traffic and lets tests inject inbound
// events through the handler it was given.
type fakeTransport struct {
	mu       sync.Mutex
	h        device.Handler
	events   []device.Event
	payloads [][]byte
	audio    [][]byte
}

func (f *fakeTransport) String() string { return "fake" }
func (f *fakeTransport) Connect(h device.Handler) error {
	f.h = h
	return nil
}
func (f *fakeTransport) Disconnect() error { return nil }
func (f *fakeTransport) SendEvent(e device.Event, p []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, e)
	f.payloads = append(f.payloads, append([]byte(nil), p...))
	return nil
}
func (f *fakeTransport) SendAudio(d []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.audio = append(f.audio, d)
	return nil
}
func (f *fakeTransport) sent() []device.Event {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]device.Event(nil), f.events...)
}
func (f *fakeTransport) lastPayload() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.payloads[len(f.payloads)-1]
}
func (f *fakeTransport) has(e device.Event) bool {
	for _, got := range f.sent() {
		if got == e {
			return true
		}
	}
	return false
}

// fakeVoice records sessions; Run drains the mic until ctx ends when
// blocking, else grabs what is queued and returns.
type fakeVoice struct {
	mu        sync.Mutex
	runs      int
	announces []string
	resets    int
	mic       [][]byte
	followUp  bool
	started   chan struct{}
	block     bool
}

func newFakeVoice() *fakeVoice { return &fakeVoice{started: make(chan struct{}, 16)} }

func (v *fakeVoice) Rate() int { return 16000 }
func (v *fakeVoice) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.resets++
}
func (v *fakeVoice) Announce(_ context.Context, _ device.Output, text string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.announces = append(v.announces, text)
}
func (v *fakeVoice) Run(ctx context.Context, _ device.Output, mic <-chan []byte, _ *agent.Env) bool {
	v.mu.Lock()
	v.runs++
	v.mu.Unlock()
	v.started <- struct{}{}
	for {
		select {
		case chunk := <-mic:
			v.mu.Lock()
			v.mic = append(v.mic, chunk)
			v.mu.Unlock()
		case <-ctx.Done():
			return false
		default:
			if !v.block {
				return v.followUp
			}
			time.Sleep(time.Millisecond)
		}
	}
}
func (v *fakeVoice) count() (runs, resets, micChunks int) {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.runs, v.resets, len(v.mic)
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("condition not met in time")
}
