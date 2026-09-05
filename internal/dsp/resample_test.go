package dsp

import (
	"math"
	"testing"
)

func sine(n int, freq, rate float64) []int16 {
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(10000 * math.Sin(2*math.Pi*freq*float64(i)/rate))
	}
	return out
}

func TestResampleLengthRatio(t *testing.T) {
	in := sine(24000, 440, 24000)

	down := Resample(in, 24000, 16000)
	up := Resample(in, 24000, 48000)

	if len(down) != 16000 || len(up) != 48000 {
		t.Fatalf("lengths = %d, %d", len(down), len(up))
	}
}

func TestResampleSameRateIsIdentity(t *testing.T) {
	in := sine(100, 440, 16000)

	out := Resample(in, 16000, 16000)

	if len(out) != len(in) || out[10] != in[10] {
		t.Fatal("same-rate resample must be identity")
	}
}

func TestResamplePreservesTone(t *testing.T) {
	in := sine(24000, 440, 24000)

	out := Resample(in, 24000, 16000)

	// Correlate against the expected 440 Hz tone at 16 kHz over the middle
	// of the buffer (away from edge effects).
	var num, den float64
	for i := 2000; i < 14000; i++ {
		ref := 10000 * math.Sin(2*math.Pi*440*float64(i)/16000)
		num += float64(out[i]) * ref
		den += ref * ref
	}
	if gain := num / den; gain < 0.9 || gain > 1.1 {
		t.Fatalf("tone not preserved, gain = %.3f", gain)
	}
}

func TestResampleEmpty(t *testing.T) {
	out := Resample(nil, 24000, 16000)

	if len(out) != 0 {
		t.Fatal("expected empty output")
	}
}

func TestPCMConversions(t *testing.T) {
	samples := []int16{-32768, 0, 32767}

	b := Int16ToBytes(samples)
	back := BytesToInt16(b)
	f := BytesToFloat32(b)

	if len(b) != 6 || back[0] != -32768 || back[2] != 32767 {
		t.Fatalf("round trip failed: %v", back)
	}
	if f[0] != -1 || f[1] != 0 || f[2] < 0.999 {
		t.Fatalf("float conversion: %v", f)
	}
}

func TestFloat32ToInt16Clips(t *testing.T) {
	out := Float32ToInt16([]float32{2, -2, 0.5}, 32767)

	if out[0] != 32767 || out[1] != -32768 || out[2] != 16383 {
		t.Fatalf("got %v", out)
	}
}

func TestHannWindow(t *testing.T) {
	w := HannWindow(5)

	if w[0] != 0 || w[4] != 0 || math.Abs(w[2]-1) > 1e-9 {
		t.Fatalf("got %v", w)
	}
}
