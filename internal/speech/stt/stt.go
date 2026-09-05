// Package stt turns microphone audio into text on sherpa-onnx: a Silero
// VAD finds the utterance while Nemotron (streaming) or Whisper (offline)
// recognizes it.
package stt

import (
	"context"
	"encoding/binary"
	"fmt"
	"runtime"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// SampleRate is the fixed device microphone rate.
const SampleRate = 16000

// Recognizer listens to a mic stream and returns what was said.
type Recognizer interface {
	Load() error
	// Listen consumes 16 kHz mono PCM until the user stops speaking and
	// returns the transcript ("" if nothing was said). onSpeech fires when
	// speech is first detected.
	Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error)
	Close()
}

// New builds the configured recognizer.
func New(cfg config.STTConfig) (Recognizer, error) {
	switch cfg.Provider {
	case "nemotron":
		return &nemotron{cfg: cfg}, nil
	case "whisper":
		return &whisper{cfg: cfg}, nil
	}
	return nil, fmt.Errorf("unknown STT provider %q", cfg.Provider)
}

func threads() int { return min(runtime.NumCPU(), 8) }

func errLoad(what string) error { return fmt.Errorf("sherpa-onnx failed to load %s", what) }

// samples converts little-endian 16-bit PCM to floats in [-1, 1).
func samples(pcm []byte) []float32 {
	out := make([]float32, len(pcm)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
	}
	return out
}
