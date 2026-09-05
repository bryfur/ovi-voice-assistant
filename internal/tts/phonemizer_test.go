package tts

import (
	"strings"
	"testing"
)

func fakeRun(voice string, lines []string) ([]string, error) {
	out := make([]string, len(lines))
	for i, l := range lines {
		switch strings.ToLower(l) {
		case "hello":
			out[i] = "həlˈoʊ"
		case "world":
			out[i] = "wˈɜːld"
		default:
			out[i] = "(en)t͡ʃ" + strings.ToLower(l) + "\n"
		}
	}
	return out, nil
}

func TestPhonemizePreservesPunctuation(t *testing.T) {
	p := &Phonemizer{Voice: "en-us", Run: fakeRun}

	got, err := p.Phonemize("Hello, world.")

	if err != nil || got != "həlˈoʊ, wˈɜːld." {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestPhonemizeCleansTiesAndLanguageMarkers(t *testing.T) {
	p := &Phonemizer{Voice: "en-us", Run: fakeRun}

	got, err := p.Phonemize("Xyz")

	if err != nil || got != "tʃxyz" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestPhonemizeEmpty(t *testing.T) {
	p := &Phonemizer{Voice: "en-us", Run: fakeRun}

	got, err := p.Phonemize("   ")

	if err != nil || got != "" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestPhonemizeOnlyPunctuation(t *testing.T) {
	p := &Phonemizer{Voice: "en-us", Run: fakeRun}

	got, err := p.Phonemize("...")

	if err != nil || got != "..." {
		t.Fatalf("got %q, %v", got, err)
	}
}
