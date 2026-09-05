package stt

import (
	"math"
	"os"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
)

// TestSileroVADRealModel exercises ONNX Runtime + the real Silero model.
// It downloads ~60 MB on first run, so it only runs with OVI_TEST_MODELS=1.
func TestSileroVADRealModel(t *testing.T) {
	if os.Getenv("OVI_TEST_MODELS") != "1" {
		t.Skip("set OVI_TEST_MODELS=1 to run model tests")
	}
	vad, err := LoadSileroVAD()
	if err != nil {
		t.Fatal(err)
	}
	defer vad.Close()
	st := vad.NewState()

	// Silence must score low.
	silent := make([]float32, VADChunkSamples)
	var pSilence float32
	for i := 0; i < 5; i++ {
		pSilence, err = vad.Probability(st, silent)
		if err != nil {
			t.Fatal(err)
		}
	}

	// A synthetic voiced signal (harmonics of 140 Hz with a pitch wobble
	// and amplitude envelope) should score clearly higher than silence.
	st = vad.NewState()
	var pVoice float32
	n := 0
	for i := 0; i < 30; i++ {
		chunk := make([]float32, VADChunkSamples)
		for j := range chunk {
			tt := float64(n) / 16000
			f0 := 140 + 10*math.Sin(2*math.Pi*3*tt)
			var v float64
			for h := 1; h <= 8; h++ {
				v += math.Sin(2*math.Pi*f0*float64(h)*tt) / float64(h)
			}
			env := 0.5 + 0.5*math.Sin(2*math.Pi*4*tt)
			chunk[j] = float32(0.3 * env * v)
			n++
		}
		p, err := vad.Probability(st, chunk)
		if err != nil {
			t.Fatal(err)
		}
		if p > pVoice {
			pVoice = p
		}
	}
	t.Logf("silence=%.3f voice(max)=%.3f", pSilence, pVoice)
	if pSilence > 0.3 {
		t.Fatalf("silence probability too high: %.3f", pSilence)
	}
	if pVoice <= pSilence {
		t.Fatalf("voiced signal should score above silence (%.3f vs %.3f)", pVoice, pSilence)
	}
	_ = dsp.BytesToFloat32 // keep dsp linked for the listener path
}
