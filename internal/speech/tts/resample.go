package tts

import (
	"encoding/binary"
	"math"
)

// pcm clips [-1, 1] samples to little-endian 16-bit PCM.
func pcm(samples []float32) []byte {
	out := make([]byte, 2*len(samples))
	for i, v := range samples {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(max(-32768, min(32767, v*32767)))))
	}
	return out
}

// resample converts samples between rates with an anti-aliased
// windowed-sinc interpolator. Same-rate input is returned unchanged.
func resample(in []float32, from, to int) []float32 {
	if from == to || len(in) == 0 {
		return in
	}
	ratio := float64(to) / float64(from)
	cutoff := 0.5 * min(1, ratio)
	width := int(math.Ceil(24 * max(1, 1/ratio)))
	out := make([]float32, int(math.Round(float64(len(in))*ratio)))
	for i := range out {
		center := float64(i) / ratio
		var acc, norm float64
		for j := max(int(center)-width, 0); j < min(int(center)+width+1, len(in)); j++ {
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
