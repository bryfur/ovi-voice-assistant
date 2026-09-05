// Package stt implements speech-to-text providers with Silero VAD
// end-of-speech detection.
package stt

import (
	"context"
	"fmt"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// VADStartCallback fires when VAD detects speech start.
type VADStartCallback func()

// STT is a speech-to-text engine.
type STT interface {
	// Load loads models; must be called before transcription.
	Load() error
	// Transcribe transcribes a complete 16 kHz mono PCM buffer.
	Transcribe(pcm []byte) (string, error)
	// TranscribeStream consumes mic audio until end-of-speech and returns
	// the transcript. onVADStart may be nil.
	TranscribeStream(ctx context.Context, chunks <-chan []byte, onVADStart VADStartCallback) (string, error)
}

// Create builds the configured STT provider.
func Create(settings *config.Settings) (STT, error) {
	switch settings.STT.Provider {
	case "whisper":
		return NewWhisperSTT(settings), nil
	case "nemotron":
		return NewNemotronSTT(settings), nil
	}
	return nil, fmt.Errorf("unknown STT provider: %s", settings.STT.Provider)
}
