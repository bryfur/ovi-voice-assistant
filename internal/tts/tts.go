// Package tts implements text-to-speech providers and streaming synthesis.
package tts

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// TTS is a text-to-speech engine producing 16-bit mono PCM.
type TTS interface {
	// Load loads models; must be called before synthesis.
	Load() error
	// SampleRate is the output rate in Hz (valid after Load).
	SampleRate() int
	// SampleWidth is bytes per sample (2 = 16-bit).
	SampleWidth() int
	// Channels is 1 for mono.
	Channels() int
	// Synthesize renders text to PCM in one buffer.
	Synthesize(text string) ([]byte, error)
	// SynthesizeIter renders text, calling emit for each PCM chunk as soon
	// as it is available. Providers that cannot stream emit once.
	SynthesizeIter(text string, emit func(pcm []byte) error) error
}

// Create builds the configured TTS provider at the given output rate.
func Create(settings *config.Settings, sampleRate int) (TTS, error) {
	switch settings.TTS.Provider {
	case "piper":
		return NewPiperTTS(settings, sampleRate), nil
	case "kokoro":
		return NewKokoroTTS(settings, sampleRate), nil
	}
	return nil, fmt.Errorf("unknown TTS provider: %s", settings.TTS.Provider)
}

// SynthesizeAll is a helper implementing Synthesize via SynthesizeIter.
func SynthesizeAll(t TTS, text string) ([]byte, error) {
	var out []byte
	err := t.SynthesizeIter(text, func(pcm []byte) error {
		out = append(out, pcm...)
		return nil
	})
	return out, err
}

// ListenToken is stripped from spoken output.
const ListenToken = "[LISTEN]"

const sentenceEndings = ".!?"

// SplitSentences splits a stream of text tokens into sentences at .!?
// boundaries, calling emit for each. [LISTEN] control tokens are stripped.
func SplitSentences(ctx context.Context, tokens <-chan string, emit func(sentence string) error) error {
	var buf []rune
	flush := func(sentence string) error {
		sentence = strings.TrimSpace(strings.ReplaceAll(strings.TrimSpace(sentence), ListenToken, ""))
		if sentence == "" {
			return nil
		}
		return emit(sentence)
	}
	for {
		var chunk string
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case chunk, ok = <-tokens:
		}
		if !ok {
			break
		}
		buf = append(buf, []rune(chunk)...)
		for {
			split := -1
			for i, ch := range buf {
				if strings.ContainsRune(sentenceEndings, ch) && i > 10 &&
					(i+1 >= len(buf) || buf[i+1] == ' ') {
					split = i + 1
					break
				}
			}
			if split == -1 {
				break
			}
			sentence := string(buf[:split])
			buf = []rune(strings.TrimLeft(string(buf[split:]), " \t\n\r"))
			if err := flush(sentence); err != nil {
				return err
			}
		}
	}
	return flush(string(buf))
}

// Stream synthesizes a token stream with pipelined synthesis and playback.
//
// A background producer synthesizes sentences into a small bounded channel
// while the consumer emits ready audio. The producer pulls PCM sub-chunks
// from SynthesizeIter so that providers emitting per-batch audio can feed
// the device mid-sentence.
func Stream(ctx context.Context, t TTS, tokens <-chan string, emit func(pcm []byte) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	chunks := make(chan []byte, 2)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunks)
		err := SplitSentences(ctx, tokens, func(sentence string) error {
			slog.Debug("TTS synthesizing", "text", truncate(sentence, 60))
			return t.SynthesizeIter(sentence, func(pcm []byte) error {
				if len(pcm) == 0 {
					return nil
				}
				select {
				case chunks <- pcm:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		})
		errCh <- err
	}()

	for pcm := range chunks {
		if err := emit(pcm); err != nil {
			cancel()
			for range chunks {
			}
			<-errCh
			return err
		}
	}
	if err := <-errCh; err != nil && ctx.Err() == nil {
		return err
	}
	return ctx.Err()
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
