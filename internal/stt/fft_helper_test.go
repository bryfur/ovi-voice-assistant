package stt

import "gonum.org/v1/gonum/dsp/fourier"

func newFFT() *fourier.FFT { return fourier.NewFFT(nemNFFT) }
