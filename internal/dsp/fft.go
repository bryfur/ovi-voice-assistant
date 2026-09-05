package dsp

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/dsp/fourier"
)

// PowerSpectrum computes |rfft(frame)|^2 for a real frame using a reusable
// FFT plan. The result has len(frame)/2+1 bins.
func PowerSpectrum(plan *fourier.FFT, frame []float64, coeffs []complex128, out []float64) []float64 {
	// gonum requires dst to be nil or exactly n/2+1 long.
	if want := len(frame)/2 + 1; len(coeffs) != want {
		coeffs = make([]complex128, want)
	}
	coeffs = plan.Coefficients(coeffs, frame)
	if cap(out) < len(coeffs) {
		out = make([]float64, len(coeffs))
	}
	out = out[:len(coeffs)]
	for i, c := range coeffs {
		m := cmplx.Abs(c)
		out[i] = m * m
	}
	return out
}

// HannWindow returns a symmetric Hann window of length n (matching
// numpy's np.hanning / the 0.5*(1-cos(2πi/(n-1))) form).
func HannWindow(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for i := range w {
		w[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

// HannWindowPeriodic returns a periodic Hann window (torch.hann_window default).
func HannWindowPeriodic(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(n)))
	}
	return w
}
