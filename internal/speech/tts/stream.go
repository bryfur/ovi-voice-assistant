package tts

import (
	"context"
	"log/slog"
	"strings"
)

// ListenToken marks a reply that expects a follow-up; it is never spoken.
const ListenToken = "[LISTEN]"

// Stream speaks a token stream as it arrives: sentences are rendered in
// the background while finished audio is emitted, so playback starts
// before the model has finished talking.
func Stream(ctx context.Context, s Synthesizer, tokens <-chan string, emit func(pcm []byte) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	audio := make(chan []byte, 2)
	rendered := make(chan error, 1)
	go func() {
		defer close(audio)
		rendered <- split(ctx, tokens, func(sentence string) error {
			slog.Debug("TTS", "text", sentence)
			return s.Synthesize(sentence, func(pcm []byte) error {
				select {
				case audio <- pcm:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		})
	}()
	for pcm := range audio {
		if err := emit(pcm); err != nil {
			cancel()
			for range audio {
			}
			<-rendered
			return err
		}
	}
	if err := <-rendered; err != nil && ctx.Err() == nil {
		return err
	}
	return ctx.Err()
}

// firstChunkWords is how many words the first chunk needs before it may
// end at a clause boundary.
const firstChunkWords = 4

// split turns tokens into speakable chunks, stripping ListenToken. Chunks
// end at sentence boundaries (.!?), except the very first, which may also
// end at a clause boundary (,;:) once it has a few words, so audio starts
// before the first sentence is complete.
func split(ctx context.Context, tokens <-chan string, emit func(string) error) error {
	first := true
	flush := func(s string) error {
		s = strings.TrimSpace(strings.ReplaceAll(s, ListenToken, ""))
		if s == "" {
			return nil
		}
		first = false
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
		for cut := boundary(buf, first); cut > 0; cut = boundary(buf, first) {
			chunk := string(buf[:cut])
			buf = []rune(strings.TrimLeft(string(buf[cut:]), " \t\n\r"))
			if err := flush(chunk); err != nil {
				return err
			}
		}
	}
}

// boundary returns the index just past the first chunk boundary that is
// followed by a space (or ends the buffer), or 0. Sentence boundaries
// need more than 10 characters so "3.14" or "Dr." do not split.
func boundary(buf []rune, first bool) int {
	words := 0
	for i, ch := range buf {
		if ch == ' ' {
			words++
		}
		if i+1 < len(buf) && buf[i+1] != ' ' {
			continue
		}
		if strings.ContainsRune(".!?", ch) && i > 10 {
			return i + 1
		}
		if first && strings.ContainsRune(",;:", ch) && words >= firstChunkWords-1 {
			return i + 1
		}
	}
	return 0
}
