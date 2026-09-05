package device

import (
	"bytes"
	"context"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"sync"
	"testing"
	"time"
)

type sentEvent struct {
	event   EventType
	payload []byte
}

type mockTransport struct {
	mu     sync.Mutex
	events []sentEvent
	audio  [][]byte
}

func (m *mockTransport) Connect() error                           { return nil }
func (m *mockTransport) Disconnect() error                        { return nil }
func (m *mockTransport) SetEventCallback(EventCallback)           {}
func (m *mockTransport) SetAudioCallback(AudioCallback)           {}
func (m *mockTransport) SetDisconnectCallback(DisconnectCallback) {}
func (m *mockTransport) SetConnectCallback(ConnectCallback)       {}
func (m *mockTransport) IsConnected() bool                        { return true }
func (m *mockTransport) String() string                           { return "mock" }
func (m *mockTransport) SendEvent(e EventType, p []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, sentEvent{e, append([]byte(nil), p...)})
	return nil
}
func (m *mockTransport) SendAudio(d []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.audio = append(m.audio, append([]byte(nil), d...))
	return nil
}
func (m *mockTransport) audioCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.audio)
}
func (m *mockTransport) eventList() []sentEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]sentEvent(nil), m.events...)
}

func newOutput(t *testing.T) (*EncodingOutput, *mockTransport) {
	t.Helper()
	tr := &mockTransport{}
	o := NewEncodingOutput(tr, codec.NewPCMCodec(16000, 1))
	o.Sleep = func(context.Context, time.Duration) {} // no real pacing in tests
	t.Cleanup(o.Reset)
	return o, tr
}

func TestTTSStartSendsAudioConfigFirst(t *testing.T) {
	o, tr := newOutput(t)

	err := o.SendEvent(context.Background(), EventTTSStart, nil)

	if err != nil {
		t.Fatal(err)
	}
	events := tr.eventList()
	if len(events) != 2 || events[0].event != EventAudioConfig || events[1].event != EventTTSStart {
		t.Fatalf("events = %+v", events)
	}
	cfg, _ := UnmarshalAudioConfig(events[0].payload)
	if cfg.SampleRate != 16000 || cfg.EncodedFrameBytes != 640 || cfg.CodecType != 0 || cfg.Channels != 1 {
		t.Fatalf("config = %+v", cfg)
	}
}

func TestTTSStartReportsAudioConfig(t *testing.T) {
	o, _ := newOutput(t)
	var seen []string
	o.OnAudioConfig = func(c codec.AudioCodec) { seen = append(seen, codec.Describe(c)) }

	o.SendEvent(context.Background(), EventTTSStart, nil)
	o.SendEvent(context.Background(), EventTTSEnd, nil)

	if len(seen) != 1 || seen[0] != "pcm 16000Hz 1ch 16-bit 20ms" {
		t.Fatalf("seen = %v", seen)
	}
}

func TestOtherEventsForwardedDirectly(t *testing.T) {
	o, tr := newOutput(t)

	o.SendEvent(context.Background(), EventTTSEnd, []byte{1})
	for _, ev := range []EventType{EventVADStart, EventMicStop, EventError} {
		o.SendEvent(context.Background(), ev, nil)
	}

	events := tr.eventList()
	if len(events) != 4 || events[0].event != EventTTSEnd || !bytes.Equal(events[0].payload, []byte{1}) {
		t.Fatalf("events = %+v", events)
	}
}

func TestSendAudioBuffersUntilFullFrame(t *testing.T) {
	o, tr := newOutput(t)

	o.SendAudio(context.Background(), make([]byte, 320))
	o.Flush(context.Background())

	if tr.audioCount() != 1 || len(tr.audio[0]) != 640 {
		t.Fatalf("audio = %d frames", tr.audioCount())
	}
}

func TestSendAudioMultipleFramesAndLeftover(t *testing.T) {
	o, tr := newOutput(t)

	o.SendAudio(context.Background(), make([]byte, 700))
	o.SendAudio(context.Background(), make([]byte, 580))
	o.Flush(context.Background())

	if tr.audioCount() != 2 {
		t.Fatalf("audio = %d frames", tr.audioCount())
	}
}

func TestEncodedDataMatchesInput(t *testing.T) {
	o, tr := newOutput(t)
	frame := make([]byte, 640)
	for i := range frame {
		frame[i] = byte(i)
	}

	o.SendAudio(context.Background(), frame)
	o.Flush(context.Background())

	if tr.audioCount() != 1 || !bytes.Equal(tr.audio[0], frame) {
		t.Fatal("frame mismatch")
	}
}

func TestFlushPadsPartialFrame(t *testing.T) {
	o, tr := newOutput(t)
	partial := bytes.Repeat([]byte{1}, 100)

	o.SendAudio(context.Background(), partial)
	o.Flush(context.Background())

	if tr.audioCount() != 1 || len(tr.audio[0]) != 640 {
		t.Fatalf("audio = %+v", tr.audio)
	}
	if !bytes.Equal(tr.audio[0][:100], partial) || !bytes.Equal(tr.audio[0][100:], make([]byte, 540)) {
		t.Fatal("padding wrong")
	}
	o.Flush(context.Background()) // buffer cleared
	if tr.audioCount() != 1 {
		t.Fatal("second flush must not resend")
	}
}

func TestFlushEmptyBufferNoSend(t *testing.T) {
	o, tr := newOutput(t)

	if err := o.Flush(context.Background()); err != nil || tr.audioCount() != 0 {
		t.Fatalf("err=%v frames=%d", err, tr.audioCount())
	}
}

func TestResetDiscardsBufferedAudio(t *testing.T) {
	o, tr := newOutput(t)
	o.SendAudio(context.Background(), make([]byte, 100))

	o.Reset()
	o.Flush(context.Background())

	if tr.audioCount() != 0 {
		t.Fatal("reset should discard partial buffer")
	}
}

func TestEventsOrderedAfterAudio(t *testing.T) {
	o, tr := newOutput(t)

	o.SendAudio(context.Background(), make([]byte, 640*3))
	o.SendEvent(context.Background(), EventTTSEnd, nil)

	if tr.audioCount() != 3 || len(tr.eventList()) != 1 {
		t.Fatalf("audio=%d events=%d", tr.audioCount(), len(tr.eventList()))
	}
}

func TestPacingSleepsBeyondLeadTime(t *testing.T) {
	o, _ := newOutput(t)
	now := time.Unix(0, 0)
	o.Now = func() time.Time { return now }
	var slept []time.Duration
	o.Sleep = func(_ context.Context, d time.Duration) { slept = append(slept, d) }

	// 20 frames × 20 ms = 400 ms of audio at t=0 → must sleep past the 300 ms lead.
	o.SendAudio(context.Background(), make([]byte, 640*20))
	o.Flush(context.Background())

	if len(slept) == 0 {
		t.Fatal("expected pacing sleeps")
	}
	var total time.Duration
	for _, d := range slept {
		total += d
	}
	if total < 50*time.Millisecond {
		t.Fatalf("total sleep %v too small", total)
	}
}

func TestSendEventCancelledContext(t *testing.T) {
	o, _ := newOutput(t)
	o.Sleep = func(ctx context.Context, d time.Duration) { <-ctx.Done() } // block until reset
	o.SendAudio(context.Background(), make([]byte, 640*100))              // enough to trigger a sleep
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := o.SendEvent(ctx, EventTTSEnd, nil)

	if err == nil {
		t.Fatal("expected context error")
	}
}
