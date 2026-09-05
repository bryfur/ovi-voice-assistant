package stt

import (
	"context"
	"log/slog"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
	"github.com/bryfur/ovi-voice-assistant/internal/models"
)

// Listening limits; variables so tests can shorten them.
var (
	noSpeechTimeout = 5 * time.Second  // give up if nobody talks
	maxListen       = 60 * time.Second // hard cap on one utterance
)

const (
	minSpeech    = 0.3 // seconds of speech for a valid utterance
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

// listen feeds mic audio through the VAD (and feed, if set) until a speech
// segment completes or a timeout fires. It returns the segment samples, or
// nil when nothing was said.
func listen(ctx context.Context, mic <-chan []byte, v vad, onSpeech func(), feed func([]float32)) ([]float32, error) {
	v.Reset()
	start := time.Now()
	speaking := false
	for {
		var chunk []byte
		var ok bool
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case chunk, ok = <-mic:
		}
		if !ok {
			v.Flush()
			seg, _ := v.Segment()
			return seg, nil
		}
		samples := dsp.BytesToFloat32(chunk)
		v.Accept(samples)
		if feed != nil {
			feed(samples)
		}
		if !speaking && v.Speaking() {
			speaking = true
			if onSpeech != nil {
				onSpeech()
			}
		}
		if seg, ok := v.Segment(); ok {
			slog.Info("End of speech", "secs", float64(len(seg))/SampleRate)
			return seg, nil
		}
		elapsed := time.Since(start)
		if !v.Speaking() && elapsed > noSpeechTimeout {
			slog.Info("No speech detected, giving up")
			return nil, nil
		}
		if elapsed > maxListen {
			slog.Warn("Max listen duration reached")
			v.Flush()
			seg, _ := v.Segment()
			return seg, nil
		}
	}
}

// sileroVAD adapts sherpa's VAD to the vad interface.
type sileroVAD struct{ v *sherpa.VoiceActivityDetector }

func newSileroVAD(minSilence float64) (*sileroVAD, error) {
	if minSilence <= 0 {
		minSilence = 0.75
	}
	path, err := models.EnsureFile(models.ASR, "silero_vad.onnx")
	if err != nil {
		return nil, err
	}
	cfg := sherpa.VadModelConfig{SampleRate: SampleRate, NumThreads: 1, Provider: "cpu"}
	cfg.SileroVad = sherpa.SileroVadModelConfig{
		Model:              path,
		Threshold:          vadThreshold,
		MinSilenceDuration: float32(minSilence),
		MinSpeechDuration:  minSpeech,
		MaxSpeechDuration:  float32(maxListen.Seconds()),
		WindowSize:         512,
	}
	v := sherpa.NewVoiceActivityDetector(&cfg, float32(maxListen.Seconds()))
	if v == nil {
		return nil, errLoad("Silero VAD")
	}
	return &sileroVAD{v}, nil
}

func (s *sileroVAD) Accept(samples []float32) { s.v.AcceptWaveform(samples) }
func (s *sileroVAD) Speaking() bool           { return s.v.IsSpeech() }
func (s *sileroVAD) Flush()                   { s.v.Flush() }
func (s *sileroVAD) Reset()                   { s.v.Reset() }
func (s *sileroVAD) Close()                   { sherpa.DeleteVoiceActivityDetector(s.v) }

func (s *sileroVAD) Segment() ([]float32, bool) {
	if s.v.IsEmpty() {
		return nil, false
	}
	seg := s.v.Front()
	s.v.Pop()
	return seg.Samples, true
}

type loadError string

func errLoad(what string) error { return loadError(what) }

func (e loadError) Error() string { return "sherpa-onnx failed to load " + string(e) }
