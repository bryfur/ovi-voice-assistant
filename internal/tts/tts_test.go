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

func sentences(t *testing.T, parts ...string) []string {
	t.Helper()
	var out []string
	if err := SplitSentences(context.Background(), tokensOf(parts...), func(s string) error {
		out = append(out, s)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestSplitSentences(t *testing.T) {
	cases := map[string][]string{
		"boundaries": sentences(t, "Hello there, how ", "are you today? I am ", "doing fine. Thanks!"),
		"listen":     sentences(t, "Do you want more details? [LISTEN]"),
		"short":      sentences(t, "Dr. Smith is here now."),
		"remainder":  sentences(t, "no punctuation at all"),
		"empty":      sentences(t, "   ", "[LISTEN]"),
	}
	want := map[string]string{
		"boundaries": "Hello there, how are you today?|I am doing fine.|Thanks!",
		"listen":     "Do you want more details?",
		"short":      "Dr. Smith is here now.",
		"remainder":  "no punctuation at all",
		"empty":      "",
	}
	for name, got := range cases {
		if strings.Join(got, "|") != want[name] {
			t.Errorf("%s: got %v", name, got)
		}
	}
}

// fakeTTS emits the sentence text n times per call.
type fakeTTS struct {
	n   int
	err error
}

func (f *fakeTTS) Load() error     { return nil }
func (f *fakeTTS) SampleRate() int { return 16000 }
func (f *fakeTTS) Close()          {}
func (f *fakeTTS) Synthesize(text string, emit func([]byte) error) error {
	if f.err != nil {
		return f.err
	}
	for range max(f.n, 1) {
		if err := emit([]byte(text)); err != nil {
			return err
		}
	}
	return nil
}

func TestStreamPipelinesSentences(t *testing.T) {
	var got []string

	err := Stream(context.Background(), &fakeTTS{n: 2}, tokensOf("First sentence here. ", "Second one there."), func(pcm []byte) error {
		got = append(got, string(pcm))
		return nil
	})

	if err != nil || len(got) != 4 || got[0] != "First sentence here." || got[3] != "Second one there." {
		t.Fatalf("got %v, %v", got, err)
	}
}

func TestStreamErrors(t *testing.T) {
	synthErr := Stream(context.Background(), &fakeTTS{err: errors.New("boom")}, tokensOf("Hello there friend."), func([]byte) error { return nil })
	emitErr := Stream(context.Background(), &fakeTTS{n: 3}, tokensOf("One sentence here. Another one there."), func([]byte) error { return errors.New("device gone") })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelErr := Stream(ctx, &fakeTTS{}, make(chan string), func([]byte) error { return nil })

	if synthErr == nil || !strings.Contains(synthErr.Error(), "boom") || emitErr == nil || emitErr.Error() != "device gone" || !errors.Is(cancelErr, context.Canceled) {
		t.Fatalf("synth=%v emit=%v cancel=%v", synthErr, emitErr, cancelErr)
	}
}

func TestNewProviders(t *testing.T) {
	for _, p := range []string{"kokoro", "piper"} {
		if _, err := New(config.TTSConfig{Provider: p}, 16000); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := New(config.TTSConfig{Provider: "bogus"}, 16000); err == nil {
		t.Fatal("expected error")
	}
}

func TestKokoroVoiceTable(t *testing.T) {
	if kokoroVoices["af_heart"] != 3 || kokoroVoices["bm_lewis"] != 27 || kokoroVoices["zm_yunyang"] != 52 {
		t.Fatal("voice ids wrong")
	}
	voices := KokoroVoices()
	if len(voices) != 28 || voices[0] != "af_alloy" || strings.HasPrefix(voices[len(voices)-1], "z") {
		t.Fatalf("english voices = %v", voices)
	}
}

func TestSynthesizeBeforeLoad(t *testing.T) {
	s, _ := New(config.TTSConfig{Provider: "kokoro"}, 16000)

	if err := s.Synthesize("hi", func([]byte) error { return nil }); err == nil {
		t.Fatal("expected error")
	}
}
