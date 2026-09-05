package speech

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func configFor(provider string) config.STTConfig {
	c := config.Default().STT
	c.Provider = provider
	return c
}

// voice returns n seconds of a synthetic voiced signal as PCM chunks.
func voice(seconds float64) [][]byte {
	var chunks [][]byte
	n := 0
	for len(chunks) < int(seconds*micRate/320) {
		s := make([]float32, 320)
		for i := range s {
			tt := float64(n) / micRate
			f0 := 140 + 10*math.Sin(2*math.Pi*3*tt)
			var v float64
			for h := 1; h <= 8; h++ {
				v += math.Sin(2*math.Pi*f0*float64(h)*tt) / float64(h)
			}
			s[i] = float32(0.3 * (0.5 + 0.5*math.Sin(2*math.Pi*4*tt)) * v)
			n++
		}
		chunks = append(chunks, float32ToBytes(s))
	}
	return chunks
}

// Real Silero VAD through sherpa-onnx (downloads <1 MB); OVI_TEST_MODELS=1.
func TestSileroRealEndOfSpeech(t *testing.T) {
	if os.Getenv("OVI_TEST_MODELS") != "1" {
		t.Skip("set OVI_TEST_MODELS=1")
	}
	v, err := newSileroVAD(0.5)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	mic := make(chan []byte, 1000)
	for _, c := range voice(1.5) {
		mic <- c
	}
	for range 60 { // 1.2 s of silence
		mic <- make([]byte, 640)
	}
	speech := 0

	seg, err := listen(context.Background(), mic, v, func() { speech++ }, nil)

	t.Logf("segment=%.2fs speech events=%d", float64(len(seg))/micRate, speech)
	if err != nil || speech != 1 || len(seg) < micRate/2 {
		t.Fatalf("err=%v speech=%d seg=%d", err, speech, len(seg))
	}
}

// Real Nemotron streaming recognizer (downloads ~460 MB); OVI_TEST_NEMOTRON=1.
func TestNemotronRealLoadAndListen(t *testing.T) {
	if os.Getenv("OVI_TEST_NEMOTRON") != "1" {
		t.Skip("set OVI_TEST_NEMOTRON=1")
	}
	s, err := NewSTT(configFor("nemotron"))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	t.Logf("loaded in %v", time.Since(start))
	mic := make(chan []byte, 1000)
	for _, c := range voice(1.5) {
		mic <- c
	}
	for range 60 {
		mic <- make([]byte, 640)
	}

	start = time.Now()
	text, err := s.Listen(context.Background(), mic, nil)

	t.Logf("transcript=%q in %v", text, time.Since(start))
	if err != nil {
		t.Fatal(err)
	}
}
