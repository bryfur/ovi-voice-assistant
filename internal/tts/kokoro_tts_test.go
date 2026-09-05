package tts

import (
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func TestKokoroVocabIndices(t *testing.T) {
	cases := map[rune]int64{'$': 0, ';': 1, ' ': 16, 'a': 43, 'ɑ': 69, 'ɡ': 92, 'ˈ': 156, 'ː': 158, 'ᵻ': 177}

	for r, want := range cases {
		if got := KokoroVocab[r]; got != want {
			t.Errorf("vocab[%q] = %d, want %d", r, got, want)
		}
	}
}

func TestKokoroTokenizeSkipsUnknown(t *testing.T) {
	ids := KokoroTokenize("aɑ $")

	if len(ids) != 3 || ids[0] != 43 || ids[1] != 69 || ids[2] != 0 {
		t.Fatalf("got %v", ids)
	}
}

func TestKokoroSplitPhonemesKeepsPunctuationAttached(t *testing.T) {
	got := KokoroSplitPhonemes("həlˈoʊ, wˈɜːld. bˈaɪ!")

	if len(got) != 1 || got[0] != "həlˈoʊ, wˈɜːld. bˈaɪ!" {
		t.Fatalf("got %q", got)
	}
}

func TestKokoroSplitPhonemesLongInput(t *testing.T) {
	long := strings.Repeat("ab ", 100) + ". " + strings.Repeat("cd ", 100)

	got := KokoroSplitPhonemes(long)

	if len(got) != 2 {
		t.Fatalf("expected 2 batches, got %d", len(got))
	}
	for _, b := range got {
		if len([]rune(b)) >= kokoroMaxPhonemes {
			t.Fatal("batch too long")
		}
	}
}

func TestKokoroPostProcess(t *testing.T) {
	got := kokoroPostProcess("kəkˈoːɹoʊ rʲx ɬ nˈaɪnti ~", "en-us")

	if got != "kˈoʊkəɹoʊ ɹjk l nˈaɪndi" {
		t.Fatalf("got %q", got)
	}
}

func TestKokoroNormalizeText(t *testing.T) {
	got := KokoroNormalizeText("Dr. Smith  said ‘hi’ to Mr. Jones, etc. and more")

	if got != "Doctor Smith said 'hi' to Mister Jones, etc and more" {
		t.Fatalf("got %q", got)
	}
}

func TestKokoroLangFromVoice(t *testing.T) {
	s := config.Default()
	s.TTS.Model = "bf_emma"
	k := NewKokoroTTS(s, 0)

	if k.lang() != "en-gb" || k.SampleRate() != kokoroNativeRate {
		t.Fatalf("lang=%s rate=%d", k.lang(), k.SampleRate())
	}
	s.TTS.Model = ""
	if k.voice() != "af_heart" || k.lang() != "en-us" {
		t.Fatal("default voice wrong")
	}
}

func TestKokoroSynthesizeBeforeLoad(t *testing.T) {
	k := NewKokoroTTS(config.Default(), 16000)

	_, err := k.Synthesize("hi")

	if err == nil {
		t.Fatal("expected error")
	}
}
