package stt

import (
	"context"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
)

// fakeVAD starts "speaking" after speakAt chunks and completes a segment at
// endAt chunks (0 = never).
type fakeVAD struct {
	accepted, speakAt, endAt int
	resets, flushes          int
}

func (f *fakeVAD) Accept(samples []float32) { f.accepted++ }
func (f *fakeVAD) Speaking() bool {
	return f.speakAt > 0 && f.accepted >= f.speakAt && (f.endAt == 0 || f.accepted < f.endAt)
}
func (f *fakeVAD) Flush() { f.flushes++ }
func (f *fakeVAD) Reset() { f.resets++ }
func (f *fakeVAD) Segment() ([]float32, bool) {
	if f.endAt > 0 && f.accepted == f.endAt {
		return make([]float32, 100), true
	}
	if f.flushes > 0 && f.accepted >= f.speakAt && f.speakAt > 0 {
		f.flushes = 0
		return make([]float32, 50), true
	}
	return nil, false
}

func feedChunks(n int) chan []byte {
	ch := make(chan []byte, n)
	for range n {
		ch <- make([]byte, 640)
	}
	return ch
}

func TestListenReturnsSegmentAndFiresSpeechOnce(t *testing.T) {
	v := &fakeVAD{speakAt: 2, endAt: 5}
	speech := 0
	var fed int

	seg, err := listen(context.Background(), feedChunks(10), v, func() { speech++ }, func(s []float32) { fed += len(s) })

	if err != nil || len(seg) != 100 || speech != 1 || v.resets != 1 || fed != 5*320 {
		t.Fatalf("seg=%d err=%v speech=%d resets=%d fed=%d", len(seg), err, speech, v.resets, fed)
	}
}

func TestListenGivesUpWithoutSpeech(t *testing.T) {
	old := noSpeechTimeout
	noSpeechTimeout = 20 * time.Millisecond
	defer func() { noSpeechTimeout = old }()
	mic := make(chan []byte)
	go func() {
		for range 20 {
			mic <- make([]byte, 640)
			time.Sleep(5 * time.Millisecond)
		}
	}()

	seg, err := listen(context.Background(), mic, &fakeVAD{}, nil, nil)

	if err != nil || seg != nil {
		t.Fatalf("seg=%v err=%v", seg, err)
	}
}

func TestListenFlushesWhenMicCloses(t *testing.T) {
	mic := feedChunks(3)
	close(mic)

	seg, err := listen(context.Background(), mic, &fakeVAD{speakAt: 1}, nil, nil)

	if err != nil || len(seg) != 50 {
		t.Fatalf("seg=%d err=%v", len(seg), err)
	}
}

func TestListenCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := listen(ctx, make(chan []byte), &fakeVAD{}, nil, nil)

	if err == nil {
		t.Fatal("expected cancellation")
	}
}

func TestNewProviders(t *testing.T) {
	for _, p := range []string{"nemotron", "whisper"} {
		if _, err := New(configFor(p)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := New(configFor("bogus")); err == nil {
		t.Fatal("expected error")
	}
	_ = dsp.BytesToFloat32
}
