package tts

import (
	"math"
	"testing"
)

func sine(n int, freq, rate float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/rate))
	}
	return out
}

func TestResampleLengthAndTone(t *testing.T) {
	in := sine(24000, 440, 24000)

	down := resample(in, 24000, 16000)
	up := resample(in, 24000, 48000)

	if len(down) != 16000 || len(up) != 48000 {
		t.Fatalf("lengths = %d, %d", len(down), len(up))
	}
	var num, den float64
	for i := 2000; i < 14000; i++ {
		ref := 0.5 * math.Sin(2*math.Pi*440*float64(i)/16000)
		num += float64(down[i]) * ref
		den += ref * ref
	}
	if gain := num / den; gain < 0.9 || gain > 1.1 {
		t.Fatalf("tone not preserved, gain = %.3f", gain)
	}
	if same := resample(in, 24000, 24000); &same[0] != &in[0] || len(resample(nil, 24000, 16000)) != 0 {
		t.Fatal("same rate must return input; empty stays empty")
	}
}

func TestPCMClips(t *testing.T) {
	b := pcm([]float32{-1, 0, 1, 2})

	if len(b) != 8 || b[0] != 0x01 || b[1] != 0x80 || b[4] != 0xff || b[5] != 0x7f || b[6] != 0xff || b[7] != 0x7f {
		t.Fatalf("pcm = %v", b)
	}
}
