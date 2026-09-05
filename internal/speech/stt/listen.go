package stt

import (
	"context"
	"log/slog"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/speech/models"
)

// Listening limits; variables so tests can shorten them.
var (
	noSpeechTimeout = 5 * time.Second  // give up if nobody starts talking
	maxListen       = 60 * time.Second // hard cap on one utterance
)

const (
	minSpeech    = 0.3 // seconds of speech that make an utterance
	vadThreshold = 0.4
)

// vad is the voice-activity surface the listen loop needs; sherpa's VAD
// satisfies it and tests use a fake.
type vad interface {
	Accept(samples []float32)
	Speaking() bool
	Segment() ([]float32, bool)
	Flush()
	Reset()
}

// listen feeds mic audio through the VAD (and feed, when given) until an
// utterance ends or a timeout fires. It returns the utterance, or nil
// when nothing was said.
func listen(ctx context.Context, mic <-chan []byte, v vad, onSpeech func(), feed func([]float32)) ([]float32, error) {
	v.Reset()
	start := time.Now()
	spoke := false
	for {
		var chunk []byte
		var ok bool
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case chunk, ok = <-mic:
		}
		if !ok {
			return flush(v), nil
		}
		s := samples(chunk)
		v.Accept(s)
		if feed != nil {
			feed(s)
		}
		if !spoke && v.Speaking() {
			spoke = true
			if onSpeech != nil {
				onSpeech()
			}
		}
		if seg, ok := v.Segment(); ok {
			slog.Info("End of speech", "secs", float64(len(seg))/SampleRate)
			return seg, nil
		}
		switch elapsed := time.Since(start); {
		case elapsed > maxListen:
			slog.Warn("Max listen duration reached")
			return flush(v), nil
		case !spoke && elapsed > noSpeechTimeout:
			slog.Info("No speech detected, giving up")
			return nil, nil
		}
	}
}

// flush closes the open utterance, if any.
func flush(v vad) []float32 {
	v.Flush()
	seg, _ := v.Segment()
	return seg
}

// silero adapts sherpa's Silero VAD to the vad interface.
type silero struct{ v *sherpa.VoiceActivityDetector }

func newSilero(silence float64) (*silero, error) {
	if silence <= 0 {
		silence = 0.75
	}
	path, err := models.EnsureFile(models.ASR, "silero_vad.onnx")
	if err != nil {
		return nil, err
	}
	cfg := sherpa.VadModelConfig{SampleRate: SampleRate, NumThreads: 1, Provider: "cpu"}
	cfg.SileroVad = sherpa.SileroVadModelConfig{
		Model:              path,
		Threshold:          vadThreshold,
		MinSilenceDuration: float32(silence),
		MinSpeechDuration:  minSpeech,
		MaxSpeechDuration:  float32(maxListen.Seconds()),
		WindowSize:         512,
	}
	v := sherpa.NewVoiceActivityDetector(&cfg, float32(maxListen.Seconds()))
	if v == nil {
		return nil, errLoad("Silero VAD")
	}
	return &silero{v}, nil
}

func (s *silero) Accept(samples []float32) { s.v.AcceptWaveform(samples) }
func (s *silero) Speaking() bool           { return s.v.IsSpeech() }
func (s *silero) Flush()                   { s.v.Flush() }
func (s *silero) Reset()                   { s.v.Reset() }
func (s *silero) Close()                   { sherpa.DeleteVoiceActivityDetector(s.v) }

func (s *silero) Segment() ([]float32, bool) {
	if s.v.IsEmpty() {
		return nil, false
	}
	seg := s.v.Front()
	s.v.Pop()
	return seg.Samples, true
}
