package pipeline

import (
	"context"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"strings"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
)

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

type fakeAgent struct {
	response string
	err      error
	inputs   []string
	resets   int
}

func (a *fakeAgent) Load() error                 { return nil }
func (a *fakeAgent) Start(context.Context) error { return nil }
func (a *fakeAgent) Stop(context.Context) error  { return nil }
func (a *fakeAgent) ResetHistory()               { a.resets++ }
func (a *fakeAgent) RunText(context.Context, string, *agent.Context) (string, error) {
	return a.response, a.err
}
func (a *fakeAgent) RunStreamed(_ context.Context, text string, _ *agent.Context, onToken func(string)) error {
	a.inputs = append(a.inputs, text)
	if a.err != nil {
		return a.err
	}
	for _, w := range strings.SplitAfter(a.response, " ") {
		onToken(w)
	}
	return nil
}

// recordingOutput captures events and audio.
type recordingOutput struct {
	mu     sync.Mutex
	audio  []string
	events []device.EventType
}

func (r *recordingOutput) SendEvent(_ context.Context, e device.EventType, _ []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
	return nil
}

func (r *recordingOutput) SendAudio(_ context.Context, pcm []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.audio = append(r.audio, string(pcm))
	return nil
}

func (r *recordingOutput) played() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.audio...)
}

func (r *recordingOutput) eventNames() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var names []string
	for _, e := range r.events {
		names = append(names, e.String())
	}
	return strings.Join(names, ",")
}
