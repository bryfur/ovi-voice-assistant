// Package speech runs speech-to-text and text-to-speech on sherpa-onnx:
// Silero VAD end-of-speech detection, Nemotron (streaming) and Whisper
// (offline) recognition, Kokoro and Piper synthesis, and sentence-level
// streaming of LLM output into audio.
package speech

import (
	"context"
	"fmt"
	"runtime"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// micRate is the fixed device microphone rate.
const micRate = 16000

// STT listens to a mic stream and returns what was said.
type STT interface {
	Load() error
	// Listen consumes 16 kHz mono PCM until the user stops speaking and
	// returns the transcript ("" if nothing was said). onSpeech fires when
	// speech is first detected.
	Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error)
	Close()
}

// NewSTT builds the configured recognizer.
func NewSTT(cfg config.STTConfig) (STT, error) {
	switch cfg.Provider {
	case "nemotron":
		return newNemotron(cfg), nil
	case "whisper":
		return newWhisper(cfg), nil
	}
	return nil, fmt.Errorf("unknown STT provider %q", cfg.Provider)
}

func threads() int { return min(runtime.NumCPU(), 8) }
