// Package stt implements speech-to-text on sherpa-onnx: a Silero VAD
// listen loop feeding Nemotron (streaming) or Whisper (offline).
package stt

import (
	"context"
	"fmt"
	"runtime"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// SampleRate is the fixed device microphone rate.
const SampleRate = 16000

// STT listens to a mic stream and returns what was said.
type STT interface {
	Load() error
	// Listen consumes 16 kHz mono PCM until the user stops speaking and
	// returns the transcript ("" if nothing was said). onSpeech fires when
	// speech is first detected.
	Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error)
	Close()
}

// New builds the configured recognizer.
func New(cfg config.STTConfig) (STT, error) {
	switch cfg.Provider {
	case "nemotron":
		return newNemotron(cfg), nil
	case "whisper":
		return newWhisper(cfg), nil
	}
	return nil, fmt.Errorf("unknown STT provider %q", cfg.Provider)
}

func threads() int { return min(runtime.NumCPU(), 8) }
