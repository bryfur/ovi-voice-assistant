package dsp

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/dsp/fourier"
)

func TestPowerSpectrumPeakAtToneBin(t *testing.T) {
	n := 512
	plan := fourier.NewFFT(n)
	frame := make([]float64, n)
	for i := range frame {
		frame[i] = math.Sin(2 * math.Pi * 32 * float64(i) / float64(n)) // bin 32
	}

	power := PowerSpectrum(plan, frame, nil, nil)

	best := 0
	for i := range power {
		if power[i] > power[best] {
			best = i
		}
	}
	if len(power) != n/2+1 || best != 32 {
		t.Fatalf("len=%d peak bin=%d", len(power), best)
	}
}
