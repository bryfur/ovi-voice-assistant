package pipeline

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

// slowTTS emits the text bytes after a small delay so ordering is observable.
type slowTTS struct {
	delay time.Duration
	mu    sync.Mutex
	calls []string
}

func (s *slowTTS) Load() error      { return nil }
func (s *slowTTS) SampleRate() int  { return 16000 }
func (s *slowTTS) SampleWidth() int { return 2 }
func (s *slowTTS) Channels() int    { return 1 }
func (s *slowTTS) Synthesize(text string) ([]byte, error) {
	return []byte(text), nil
}
func (s *slowTTS) SynthesizeIter(text string, emit func([]byte) error) error {
	s.mu.Lock()
	s.calls = append(s.calls, text)
	s.mu.Unlock()
	time.Sleep(s.delay)
	return emit([]byte(text))
}

type recordingOutput struct {
	mu     sync.Mutex
	audio  []string
	events []transport.EventType
}

func (r *recordingOutput) SendEvent(_ context.Context, e transport.EventType, _ []byte) error {
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

func TestSubmitPlaysText(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{}, out)

	err := <-q.Submit("Hello world.")
	q.Stop()

	if err != nil || len(out.played()) != 1 || out.played()[0] != "Hello world." {
		t.Fatalf("err=%v played=%v", err, out.played())
	}
}

func TestSubmitReturnsBeforePlayback(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 50 * time.Millisecond}, out)

	start := time.Now()
	done := q.Submit("Slow sentence here.")
	elapsed := time.Since(start)
	<-done
	q.Stop()

	if elapsed > 40*time.Millisecond {
		t.Fatalf("Submit blocked for %v", elapsed)
	}
}

func TestSubmissionsPlayInOrder(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 5 * time.Millisecond}, out)

	q.Submit("First sentence here.")
	q.Submit("Second sentence here.")
	q.Submit("Third sentence here.")
	q.Stop()

	got := out.played()
	if len(got) != 3 || got[0] != "First sentence here." || got[2] != "Third sentence here." {
		t.Fatalf("played = %v", got)
	}
}

func TestStopDrainsWorkerAndIsIdempotent(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 10 * time.Millisecond}, out)
	q.Submit("Something to say here.")

	q.Stop()
	q.Stop()

	if len(out.played()) != 1 {
		t.Fatal("Stop must wait for queued speech")
	}
	if err := <-q.Submit("after stop"); err == nil {
		t.Fatal("submissions after Stop should fail")
	}
}
