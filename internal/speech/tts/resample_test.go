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
}

func TestResampleIdentityAndEmpty(t *testing.T) {
	in := sine(10, 440, 16000)

	if out := resample(in, 16000, 16000); &out[0] != &in[0] {
		t.Fatal("same rate must return input")
	}
	if out := resample(nil, 24000, 16000); len(out) != 0 {
		t.Fatal("expected empty")
	}
}

func TestPCMConversions(t *testing.T) {
	pcm := float32ToBytes([]float32{-1, 0, 1, 2})

	if len(pcm) != 8 || pcm[0] != 0x01 || pcm[1] != 0x80 || pcm[4] != 0xff || pcm[5] != 0x7f {
		t.Fatalf("pcm = %v", pcm)
	}
}
