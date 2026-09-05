package stt

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/llm"
)

// Transcriber sends audio to a transcription endpoint.
type Transcriber interface {
	Transcribe(ctx context.Context, wav []byte, model, language string) (string, error)
}

// WhisperSTT is speech-to-text using Silero VAD for end-of-speech and an
// OpenAI-compatible /audio/transcriptions endpoint (OpenAI, faster-whisper
// server, whisper.cpp server, LocalAI, ...) for recognition.
type WhisperSTT struct {
	settings *config.Settings
	vad      Prober
	client   Transcriber
	params   ListenParams

	// LoadVAD constructs the VAD; tests may override it.
	LoadVAD func() (Prober, error)
}

// NewWhisperSTT creates an unloaded provider.
func NewWhisperSTT(settings *config.Settings) *WhisperSTT {
	return &WhisperSTT{
		settings: settings,
		params:   DefaultListenParams(settings.Mic.SampleRate, settings.Mic.SampleWidth),
		LoadVAD: func() (Prober, error) {
			return LoadSileroVAD()
		},
	}
}

// Load initialises the VAD and API client.
func (w *WhisperSTT) Load() error {
	baseURL := w.settings.STT.BaseURL
	if baseURL == "" {
		baseURL = w.settings.LLM.BaseURL
	}
	apiKey := w.settings.STT.APIKey
	if apiKey == "" {
		apiKey = w.settings.LLM.APIKey
	}
	if w.client == nil {
		w.client = llm.New(baseURL, apiKey)
	}
	slog.Info("Whisper STT via transcription API", "model", w.settings.STT.Model, "base_url", baseURL)
	vad, err := w.LoadVAD()
	if err != nil {
		return err
	}
	w.vad = vad
	slog.Info("STT ready (with Silero VAD)")
	return nil
}

// SetClient overrides the transcription client (for tests).
func (w *WhisperSTT) SetClient(c Transcriber) { w.client = c }

// Transcribe sends a PCM buffer to the transcription endpoint.
func (w *WhisperSTT) Transcribe(pcm []byte) (string, error) {
	return w.transcribe(context.Background(), pcm)
}

func (w *WhisperSTT) transcribe(ctx context.Context, pcm []byte) (string, error) {
	if w.client == nil {
		return "", fmt.Errorf("call Load() first")
	}
	minBytes := w.settings.Mic.SampleRate * w.settings.Mic.SampleWidth / 10
	if len(pcm) < minBytes {
		return "", nil
	}
	wav := WAV(pcm, w.settings.Mic.SampleRate, w.settings.Mic.Channels)
	text, err := w.client.Transcribe(ctx, wav, w.settings.STT.Model, w.settings.STT.Language)
	if err != nil {
		return "", err
	}
	slog.Debug("Transcribed", "text", text)
	return text, nil
}

// TranscribeStream collects audio with Silero VAD to detect end-of-speech,
// then transcribes.
func (w *WhisperSTT) TranscribeStream(ctx context.Context, chunks <-chan []byte, onVADStart VADStartCallback) (string, error) {
	if w.vad == nil {
		return "", fmt.Errorf("call Load() first")
	}
	listener := NewVADListener(w.vad, w.params, onVADStart)
	var buf bytes.Buffer
loop:
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case chunk, ok := <-chunks:
			if !ok {
				break loop
			}
			res, err := listener.Feed(chunk)
			if err != nil {
				return "", err
			}
			switch res {
			case FeedGiveUp:
				break loop
			case FeedEndOfSpeech:
				buf.Write(chunk)
				break loop
			case FeedResetSpeech:
				buf.Reset()
			default:
				buf.Write(chunk)
			}
		}
	}
	if buf.Len() == 0 {
		return "", nil
	}
	return w.transcribe(ctx, buf.Bytes())
}

// WAV wraps 16-bit PCM in a RIFF/WAVE container.
func WAV(pcm []byte, sampleRate, channels int) []byte {
	if channels <= 0 {
		channels = 1
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+len(pcm)))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&b, binary.LittleEndian, uint32(sampleRate*channels*2))
	binary.Write(&b, binary.LittleEndian, uint16(channels*2))
	binary.Write(&b, binary.LittleEndian, uint16(16))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}
