package stt

import (
	"math"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func TestPreEmphasis(t *testing.T) {
	out := preEmphasis([]float64{1, 1, 1}, 0)

	if out[0] != 1 || math.Abs(out[1]-0.03) > 1e-9 || math.Abs(out[2]-0.03) > 1e-9 {
		t.Fatalf("got %v", out)
	}
	carried := preEmphasis([]float64{1}, 1)
	if math.Abs(carried[0]-0.03) > 1e-9 {
		t.Fatalf("carried prev wrong: %v", carried)
	}
}

func TestTokensToText(t *testing.T) {
	n := &NemotronSTT{tokens: []string{"▁hel", "lo", "▁world", "<blk>"}}

	got := n.tokensToText([]int64{0, 1, 2, 99})

	if got != "hello world" {
		t.Fatalf("got %q", got)
	}
	if n.tokensToText(nil) != "" {
		t.Fatal("empty ids should give empty text")
	}
}

func TestArgmax(t *testing.T) {
	if argmax([]float32{0.1, 0.9, 0.5}) != 1 || argmax([]float32{3}) != 0 {
		t.Fatal("argmax wrong")
	}
}

func TestToRowMajor(t *testing.T) {
	frames := make([]float32, 2*nemMels)
	frames[0] = 1         // frame 0, mel 0
	frames[nemMels+1] = 2 // frame 1, mel 1

	rm := toRowMajor(frames, 2)

	if rm[0*2+0] != 1 || rm[1*2+1] != 2 {
		t.Fatal("row-major transpose wrong")
	}
}

func TestNemotronTranscribeBeforeLoad(t *testing.T) {
	n := NewNemotronSTT(config.Default())

	_, err := n.Transcribe(make([]byte, 32000))

	if err == nil {
		t.Fatal("expected error before Load")
	}
	if n.params.SilenceTimeout.Milliseconds() != 750 {
		t.Fatal("nemotron silence timeout should be 750ms")
	}
}

func TestMelFramesShapeAndLogFloor(t *testing.T) {
	n := &NemotronSTT{}
	n.hann = make([]float64, nemNFFT)
	n.fb = make([]float64, nemMels*(nemNFFT/2+1))
	n.fftPool.New = func() any { return newFFT() }
	audio := make([]float64, 16000)

	mel := n.melFrames(audio, 0, 3)

	if len(mel) != 3*nemMels {
		t.Fatalf("len = %d", len(mel))
	}
	want := float32(math.Log(nemLogGuard))
	if mel[0] != want {
		t.Fatalf("silence should hit the log floor: %v vs %v", mel[0], want)
	}
}
