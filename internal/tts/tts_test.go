package tts

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func tokensOf(parts ...string) <-chan string {
	ch := make(chan string, len(parts))
	for _, p := range parts {
		ch <- p
	}
	close(ch)
	return ch
}

func collectSentences(t *testing.T, parts ...string) []string {
	t.Helper()
	var out []string
	err := SplitSentences(context.Background(), tokensOf(parts...), func(s string) error {
		out = append(out, s)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestSplitSentencesAtBoundaries(t *testing.T) {
	got := collectSentences(t, "Hello there, how ", "are you today? I am ", "doing fine. Thanks!")

	want := []string{"Hello there, how are you today?", "I am doing fine.", "Thanks!"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %v", got)
	}
}

func TestSplitSentencesStripsListenToken(t *testing.T) {
	got := collectSentences(t, "Do you want more details? [LISTEN]")

	if len(got) != 1 || got[0] != "Do you want more details?" {
		t.Fatalf("got %v", got)
	}
}

func TestSplitSentencesShortFragmentsNotSplit(t *testing.T) {
	got := collectSentences(t, "Dr. Smith is here now.")

	if len(got) != 1 || got[0] != "Dr. Smith is here now." {
		t.Fatalf("got %v", got)
	}
}

func TestSplitSentencesRemainderAndEmpty(t *testing.T) {
	if got := collectSentences(t, "no punctuation at all"); len(got) != 1 {
		t.Fatalf("got %v", got)
	}
	if got := collectSentences(t, "   ", "[LISTEN]"); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
}

// fakeTTS emits one chunk per sentence containing the sentence bytes.
type fakeTTS struct {
	rate   int
	err    error
	calls  []string
	chunks int
}

func (f *fakeTTS) Load() error      { return nil }
func (f *fakeTTS) SampleRate() int  { return f.rate }
func (f *fakeTTS) SampleWidth() int { return 2 }
func (f *fakeTTS) Channels() int    { return 1 }
func (f *fakeTTS) Synthesize(text string) ([]byte, error) {
	return SynthesizeAll(f, text)
}
func (f *fakeTTS) SynthesizeIter(text string, emit func([]byte) error) error {
	f.calls = append(f.calls, text)
	if f.err != nil {
		return f.err
	}
	n := f.chunks
	if n == 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		if err := emit([]byte(text)); err != nil {
			return err
		}
	}
	return nil
}

func TestStreamPipelinesSentences(t *testing.T) {
	f := &fakeTTS{rate: 16000, chunks: 2}
	var got []string

	err := Stream(context.Background(), f, tokensOf("First sentence here. ", "Second one there."), func(pcm []byte) error {
		got = append(got, string(pcm))
		return nil
	})

	if err != nil || len(got) != 4 || got[0] != "First sentence here." || got[3] != "Second one there." {
		t.Fatalf("got %v, %v", got, err)
	}
}

func TestStreamPropagatesSynthesisError(t *testing.T) {
	f := &fakeTTS{rate: 16000, err: errors.New("boom")}

	err := Stream(context.Background(), f, tokensOf("Hello there friend."), func([]byte) error { return nil })

	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("got %v", err)
	}
}

func TestStreamStopsOnEmitError(t *testing.T) {
	f := &fakeTTS{rate: 16000, chunks: 3}

	err := Stream(context.Background(), f, tokensOf("One sentence here. Another one there."), func([]byte) error { return errors.New("device gone") })

	if err == nil || err.Error() != "device gone" {
		t.Fatalf("got %v", err)
	}
}

func TestStreamCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	f := &fakeTTS{rate: 16000}

	err := Stream(ctx, f, make(chan string), func([]byte) error { return nil })

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestCreateProviders(t *testing.T) {
	s := config.Default()
	for _, p := range []string{"kokoro", "piper"} {
		s.TTS.Provider = p
		if _, err := Create(s, 16000); err != nil {
			t.Fatal(err)
		}
	}
	s.TTS.Provider = "bogus"

	_, err := Create(s, 16000)

	if err == nil {
		t.Fatal("expected error")
	}
}
