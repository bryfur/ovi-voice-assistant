package stt

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// voice returns seconds of a synthetic voiced signal as 20 ms PCM chunks.
func voice(seconds float64) [][]byte {
	var out [][]byte
	n := 0
	for range int(seconds * SampleRate / 320) {
		chunk := make([]byte, 640)
		for i := range 320 {
			t := float64(n) / SampleRate
			f0 := 140 + 10*math.Sin(2*math.Pi*3*t)
			var v float64
			for h := 1; h <= 8; h++ {
				v += math.Sin(2*math.Pi*f0*float64(h)*t) / float64(h)
			}
			v *= 0.3 * (0.5 + 0.5*math.Sin(2*math.Pi*4*t))
			binary.LittleEndian.PutUint16(chunk[2*i:], uint16(int16(v*32767)))
			n++
		}
		out = append(out, chunk)
	}
	return out
}

func micWith(chunks [][]byte, silence int) chan []byte {
	mic := make(chan []byte, len(chunks)+silence)
	for _, c := range chunks {
		mic <- c
	}
	for range silence {
		mic <- make([]byte, 640)
	}
	return mic
}

// Real Silero VAD through sherpa-onnx (downloads <1 MB); OVI_TEST_MODELS=1.
func TestSileroRealEndOfSpeech(t *testing.T) {
	if os.Getenv("OVI_TEST_MODELS") != "1" {
		t.Skip("set OVI_TEST_MODELS=1")
	}
	v, err := newSilero(0.5)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	speech := 0

	seg, err := listen(context.Background(), micWith(voice(1.5), 60), v, func() { speech++ }, nil)

	t.Logf("segment=%.2fs speech events=%d", float64(len(seg))/SampleRate, speech)
	if err != nil || speech != 1 || len(seg) < SampleRate/2 {
		t.Fatalf("err=%v speech=%d seg=%d", err, speech, len(seg))
	}
}

// Real Nemotron streaming recognizer (downloads ~460 MB); OVI_TEST_NEMOTRON=1.
func TestNemotronRealLoadAndListen(t *testing.T) {
	if os.Getenv("OVI_TEST_NEMOTRON") != "1" {
		t.Skip("set OVI_TEST_NEMOTRON=1")
	}
	s, _ := New(config.Default().STT)
	start := time.Now()
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	t.Logf("loaded in %v", time.Since(start))

	start = time.Now()
	text, err := s.Listen(context.Background(), micWith(voice(1.5), 60), nil)

	t.Logf("transcript=%q in %v", text, time.Since(start))
	if err != nil {
		t.Fatal(err)
	}
}
