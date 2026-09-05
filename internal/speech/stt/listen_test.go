package stt

import (
	"context"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// fakeVAD starts "speaking" after speakAt chunks and completes a segment
// at endAt chunks (0 = never).
type fakeVAD struct {
	accepted, speakAt, endAt int
	resets, flushes          int
}

func (f *fakeVAD) Accept([]float32) { f.accepted++ }
func (f *fakeVAD) Speaking() bool {
	return f.speakAt > 0 && f.accepted >= f.speakAt && (f.endAt == 0 || f.accepted < f.endAt)
}
func (f *fakeVAD) Flush() { f.flushes++ }
func (f *fakeVAD) Reset() { f.resets++ }
func (f *fakeVAD) Segment() ([]float32, bool) {
	if f.endAt > 0 && f.accepted == f.endAt {
		return make([]float32, 100), true
	}
	if f.flushes > 0 && f.speakAt > 0 && f.accepted >= f.speakAt {
		f.flushes = 0
		return make([]float32, 50), true
	}
	return nil, false
}

func chunks(n int) chan []byte {
	ch := make(chan []byte, n)
	for range n {
		ch <- make([]byte, 640)
	}
	return ch
}

func TestListenReturnsSegmentAndFiresSpeechOnce(t *testing.T) {
	v := &fakeVAD{speakAt: 2, endAt: 5}
	speech, fed := 0, 0

	seg, err := listen(context.Background(), chunks(10), v, func() { speech++ }, func(s []float32) { fed += len(s) })

	if err != nil || len(seg) != 100 || speech != 1 || v.resets != 1 || fed != 5*320 {
		t.Fatalf("seg=%d err=%v speech=%d resets=%d fed=%d", len(seg), err, speech, v.resets, fed)
	}
}

func TestListenGivesUpOnlyIfNobodySpoke(t *testing.T) {
	old := noSpeechTimeout
	noSpeechTimeout = 20 * time.Millisecond
	defer func() { noSpeechTimeout = old }()
	slow := func(n int) chan []byte {
		mic := make(chan []byte)
		go func() {
			for range n {
				mic <- make([]byte, 640)
				time.Sleep(5 * time.Millisecond)
			}
			close(mic)
		}()
		return mic
	}

	silent, err := listen(context.Background(), slow(20), &fakeVAD{}, nil, nil)
	late, err2 := listen(context.Background(), slow(20), &fakeVAD{speakAt: 2, endAt: 15}, nil, nil) // ends after the timeout

	if err != nil || silent != nil || err2 != nil || len(late) != 100 {
		t.Fatalf("silent=%v err=%v late=%d err2=%v", silent, err, len(late), err2)
	}
}

func TestListenFlushesWhenMicClosesAndHonoursCancel(t *testing.T) {
	mic := chunks(3)
	close(mic)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	seg, err := listen(context.Background(), mic, &fakeVAD{speakAt: 1}, nil, nil)
	_, cerr := listen(cancelled, make(chan []byte), &fakeVAD{}, nil, nil)

	if err != nil || len(seg) != 50 || cerr == nil {
		t.Fatalf("seg=%d err=%v cerr=%v", len(seg), err, cerr)
	}
}

func TestNewProvidersAndSamples(t *testing.T) {
	for _, p := range []string{"nemotron", "whisper"} {
		if _, err := New(config.STTConfig{Provider: p}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := New(config.STTConfig{Provider: "bogus"}); err == nil {
		t.Fatal("expected error")
	}
	if s := samples([]byte{0x00, 0x80, 0xff, 0x7f}); s[0] != -1 || s[1] < 0.999 {
		t.Fatalf("samples = %v", s)
	}
}
