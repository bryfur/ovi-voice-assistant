// Package tts implements text-to-speech on sherpa-onnx with streaming
// sentence delivery.
package tts

import (
	"context"
	"log/slog"
	"strings"
)

// TTS renders text to 16-bit mono PCM at SampleRate.
type TTS interface {
	Load() error
	SampleRate() int
	// Synthesize renders text, calling emit with PCM as each sentence is ready.
	Synthesize(text string, emit func(pcm []byte) error) error
	Close()
}

// ListenToken marks a response that expects a follow-up; it is never spoken.
const ListenToken = "[LISTEN]"

// SplitSentences turns a token stream into sentences at .!? boundaries,
// stripping [LISTEN], and calls emit for each.
func SplitSentences(ctx context.Context, tokens <-chan string, emit func(string) error) error {
	flush := func(s string) error {
		s = strings.TrimSpace(strings.ReplaceAll(s, ListenToken, ""))
		if s == "" {
			return nil
		}
		return emit(s)
	}
	var buf []rune
	for {
		var tok string
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case tok, ok = <-tokens:
		}
		if !ok {
			return flush(string(buf))
		}
		buf = append(buf, []rune(tok)...)
		for {
			cut := sentenceEnd(buf)
			if cut < 0 {
				break
			}
			sentence := string(buf[:cut])
			buf = []rune(strings.TrimLeft(string(buf[cut:]), " \t\n\r"))
			if err := flush(sentence); err != nil {
				return err
			}
		}
	}
}

// sentenceEnd returns the index just past the first sentence terminator
// that is followed by a space (or ends the buffer), or -1.
func sentenceEnd(buf []rune) int {
	for i, ch := range buf {
		if strings.ContainsRune(".!?", ch) && i > 10 && (i+1 == len(buf) || buf[i+1] == ' ') {
			return i + 1
		}
	}
	return -1
}

// Stream synthesizes a token stream with pipelined synthesis and playback:
// a producer renders sentences into a small channel while the consumer
// emits finished audio.
func Stream(ctx context.Context, t TTS, tokens <-chan string, emit func(pcm []byte) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	chunks := make(chan []byte, 2)
	errc := make(chan error, 1)
	go func() {
		defer close(chunks)
		errc <- SplitSentences(ctx, tokens, func(sentence string) error {
			slog.Debug("TTS", "text", sentence)
			return t.Synthesize(sentence, func(pcm []byte) error {
				select {
				case chunks <- pcm:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		})
	}()
	for pcm := range chunks {
		if err := emit(pcm); err != nil {
			cancel()
			for range chunks {
			}
			<-errc
			return err
		}
	}
	if err := <-errc; err != nil && ctx.Err() == nil {
		return err
	}
	return ctx.Err()
}
