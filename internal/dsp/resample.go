// Package dsp holds small signal-processing helpers: resampling, PCM
// conversion, and FFT-based spectrogram utilities.
package dsp

import (
	"encoding/binary"
	"math"
)

// BytesToInt16 converts little-endian 16-bit PCM bytes to samples.
func BytesToInt16(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

// Int16ToBytes converts samples to little-endian 16-bit PCM bytes.
func Int16ToBytes(s []int16) []byte {
	out := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}

// Int16ToFloat32 scales samples to [-1, 1).
func Int16ToFloat32(s []int16) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = float32(v) / 32768.0
	}
	return out
}

// BytesToFloat32 converts PCM bytes to float32 samples in [-1, 1).
func BytesToFloat32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(b[i*2:]))) / 32768.0
	}
	return out
}

// Float32ToInt16 clips and scales [-1, 1] samples to int16.
func Float32ToInt16(f []float32, gain float32) []int16 {
	out := make([]int16, len(f))
	for i, v := range f {
		x := v * gain
		if x > 32767 {
			x = 32767
		} else if x < -32768 {
			x = -32768
		}
		out[i] = int16(x)
	}
	return out
}

// Resample converts 16-bit mono PCM between sample rates using a windowed
// sinc low-pass interpolator (anti-aliased).
func Resample(audio []int16, srcRate, dstRate int) []int16 {
	if srcRate == dstRate || len(audio) == 0 {
		return audio
	}
	ratio := float64(dstRate) / float64(srcRate)
	outLen := int(math.Round(float64(len(audio)) * ratio))
	if outLen == 0 {
		return nil
	}
	// Cutoff at the lower Nyquist frequency, expressed relative to the
	// source sample rate.
	cutoff := 0.5 * math.Min(1.0, ratio)
	const halfTaps = 24
	// When downsampling the filter must be stretched to cover more
	// source samples.
	stretch := 1.0
	if ratio < 1 {
		stretch = 1 / ratio
	}
	width := int(math.Ceil(halfTaps * stretch))
	out := make([]int16, outLen)
	for i := 0; i < outLen; i++ {
		center := float64(i) / ratio
		start := int(math.Floor(center)) - width
		end := int(math.Floor(center)) + width + 1
		var acc, norm float64
		for j := start; j < end; j++ {
			if j < 0 || j >= len(audio) {
				continue
			}
			x := (float64(j) - center)
			w := sinc(2*cutoff*x) * 2 * cutoff * hann(x/float64(width+1))
			acc += float64(audio[j]) * w
			norm += w
		}
		if norm != 0 {
			acc /= norm
		}
		if acc > 32767 {
			acc = 32767
		} else if acc < -32768 {
			acc = -32768
		}
		out[i] = int16(math.Round(acc))
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

func hann(t float64) float64 {
	if t <= -1 || t >= 1 {
		return 0
	}
	return 0.5 * (1 + math.Cos(math.Pi*t))
}

// ResampleBytes is Resample for raw PCM byte buffers.
func ResampleBytes(pcm []byte, srcRate, dstRate int) []byte {
	if srcRate == dstRate {
		return pcm
	}
	return Int16ToBytes(Resample(BytesToInt16(pcm), srcRate, dstRate))
}
