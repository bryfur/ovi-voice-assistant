package device

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
)

// mockTransport records what a Speaker sends.
type mockTransport struct {
	mu     sync.Mutex
	events []Event
	frames [][]byte
}

func (m *mockTransport) String() string        { return "mock" }
func (m *mockTransport) Connect(Handler) error { return nil }
func (m *mockTransport) Disconnect() error     { return nil }
func (m *mockTransport) SendEvent(e Event, _ []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, e)
	return nil
}
func (m *mockTransport) SendAudio(f []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.frames = append(m.frames, append([]byte(nil), f...))
	return nil
}
func (m *mockTransport) sent() ([]Event, [][]byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Event(nil), m.events...), append([][]byte(nil), m.frames...)
}

func newSpeaker(t *testing.T) (*Speaker, *mockTransport) {
	t.Helper()
	tr := &mockTransport{}
	c, _ := codec.New("pcm", 16000, 1, 0)
	s := NewSpeaker(tr, c)
	s.sleep = func(context.Context, time.Duration) {}
	t.Cleanup(s.Reset)
	return s, tr
}

var ctx = context.Background()

func TestTTSStartAnnouncesFormatFirst(t *testing.T) {
	s, tr := newSpeaker(t)
	var seen []codec.Format
	s.OnConfig = func(f codec.Format) { seen = append(seen, f) }

	s.SendEvent(ctx, EventTTSStart, nil)
	s.SendEvent(ctx, EventTTSEnd, nil)

	events, _ := tr.sent()
	if len(events) != 3 || events[0] != EventAudioConfig || events[1] != EventTTSStart || len(seen) != 1 {
		t.Fatalf("events = %v seen = %v", events, seen)
	}
}

func TestAudioIsFramedAndPadded(t *testing.T) {
	s, tr := newSpeaker(t)
	frame := bytes.Repeat([]byte{1}, 640)

	s.SendAudio(ctx, frame[:320])
	s.SendAudio(ctx, frame[:320]) // completes frame 1
	s.SendAudio(ctx, frame[:100]) // partial
	s.Flush(ctx)

	_, frames := tr.sent()
	if len(frames) != 2 || !bytes.Equal(frames[0], frame) {
		t.Fatalf("frames = %d", len(frames))
	}
	if !bytes.Equal(frames[1][:100], frame[:100]) || !bytes.Equal(frames[1][100:], make([]byte, 540)) {
		t.Fatal("tail not padded")
	}
	s.Flush(ctx)
	if _, again := tr.sent(); len(again) != 2 {
		t.Fatal("flush must not resend")
	}
}

func TestEventsQueueBehindAudio(t *testing.T) {
	s, tr := newSpeaker(t)

	s.SendAudio(ctx, make([]byte, 640*3))
	s.SendEvent(ctx, EventTTSEnd, nil)

	events, frames := tr.sent()
	if len(frames) != 3 || len(events) != 1 {
		t.Fatalf("frames=%d events=%v", len(frames), events)
	}
}

func TestPacingSleepsPastLead(t *testing.T) {
	s, _ := newSpeaker(t)
	now := time.Unix(0, 0)
	s.now = func() time.Time { return now }
	var slept time.Duration
	s.sleep = func(_ context.Context, d time.Duration) { slept += d }

	s.SendAudio(ctx, make([]byte, 640*20)) // 400 ms of audio at t=0
	s.Flush(ctx)

	if slept < 50*time.Millisecond {
		t.Fatalf("slept %v, expected pacing past the %v lead", slept, lead)
	}
}

func TestResetDiscardsAndUnblocks(t *testing.T) {
	s, tr := newSpeaker(t)
	s.sleep = func(ctx context.Context, _ time.Duration) { <-ctx.Done() }
	s.SendAudio(ctx, make([]byte, 640*100)) // enough to block in a sleep
	s.SendAudio(ctx, make([]byte, 100))
	errc := make(chan error, 1)
	go func() { errc <- s.SendEvent(ctx, EventTTSEnd, nil) }()

	time.Sleep(20 * time.Millisecond)
	s.Reset()

	if err := <-errc; err != errReset {
		t.Fatalf("waiter got %v", err)
	}
	s.Flush(ctx)
	if _, frames := tr.sent(); len(frames) > 30 {
		t.Fatalf("partial buffer or queue survived reset: %d frames", len(frames))
	}
}

func TestSendEventHonoursContext(t *testing.T) {
	s, _ := newSpeaker(t)
	s.sleep = func(ctx context.Context, _ time.Duration) { <-ctx.Done() }
	s.SendAudio(ctx, make([]byte, 640*100))
	c, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()

	if err := s.SendEvent(c, EventTTSEnd, nil); err == nil {
		t.Fatal("expected context error")
	}
}
