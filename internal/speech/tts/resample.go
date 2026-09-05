package tts

import (
	"encoding/binary"
	"math"
)

// float32ToBytes clips [-1, 1] samples to little-endian 16-bit PCM.
func float32ToBytes(f []float32) []byte {
	out := make([]byte, len(f)*2)
	for i, v := range f {
		x := v * 32767
		if x > 32767 {
			x = 32767
		} else if x < -32768 {
			x = -32768
		}
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(x)))
	}
	return out
}

// resample converts float samples between rates with a windowed-sinc
// interpolator (anti-aliased). Same-rate input is returned unchanged.
func resample(in []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(in) == 0 {
		return in
	}
	ratio := float64(dstRate) / float64(srcRate)
	n := int(math.Round(float64(len(in)) * ratio))
	cutoff := 0.5 * math.Min(1, ratio)
	width := int(math.Ceil(24 * math.Max(1, 1/ratio)))
	out := make([]float32, n)
	for i := range out {
		center := float64(i) / ratio
		lo, hi := int(center)-width, int(center)+width+1
		var acc, norm float64
		for j := max(lo, 0); j < min(hi, len(in)); j++ {
			x := float64(j) - center
			w := sinc(2*cutoff*x) * hann(x/float64(width+1))
			acc += float64(in[j]) * w
			norm += w
		}
		if norm != 0 {
			acc /= norm
		}
		out[i] = float32(acc)
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

func hann(t float64) float64 {
	if t <= -1 || t >= 1 {
		return 0
	}
	return 0.5 * (1 + math.Cos(math.Pi*t))
}
