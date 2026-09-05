package tts

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// EspeakPath is the espeak-ng binary used for phonemization.
var EspeakPath = "espeak-ng"

// ErrEspeakMissing is returned when espeak-ng is not installed.
var ErrEspeakMissing = errors.New("espeak-ng not found in PATH — install it " +
	"(pacman -S espeak-ng / apt install espeak-ng / brew install espeak-ng)")

// CheckEspeak verifies espeak-ng is available.
func CheckEspeak() error {
	if _, err := exec.LookPath(EspeakPath); err != nil {
		return ErrEspeakMissing
	}
	return nil
}

var (
	langSwitchRe = regexp.MustCompile(`\([a-zA-Z\-]+\)`)
	punctSplitRe = regexp.MustCompile(`([.,!?;:])`)
	spaceRe      = regexp.MustCompile(`\s+`)
)

// Phonemizer converts text to IPA phonemes with espeak-ng.
type Phonemizer struct {
	Voice string // espeak voice, e.g. "en-us"
	// Run executes espeak on a batch of lines; tests may override it.
	Run func(voice string, lines []string) ([]string, error)
}

// NewPhonemizer creates a phonemizer for an espeak voice.
func NewPhonemizer(voice string) *Phonemizer {
	return &Phonemizer{Voice: voice, Run: runEspeak}
}

// runEspeak phonemizes each line via one espeak-ng invocation reading
// stdin. Falls back to one process per line if line counts disagree.
func runEspeak(voice string, lines []string) ([]string, error) {
	out, err := espeakStdin(voice, strings.Join(lines, "\n"))
	if err == nil {
		got := strings.Split(strings.TrimRight(out, "\n"), "\n")
		if len(got) == len(lines) {
			return got, nil
		}
	}
	res := make([]string, len(lines))
	for i, line := range lines {
		o, err := espeakStdin(voice, line)
		if err != nil {
			return nil, err
		}
		res[i] = strings.ReplaceAll(strings.TrimSpace(o), "\n", " ")
	}
	return res, nil
}

func espeakStdin(voice, text string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, EspeakPath, "-q", "--ipa", "-v", voice, "--stdin")
	cmd.Stdin = strings.NewReader(text)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return "", ErrEspeakMissing
		}
		return "", fmt.Errorf("espeak-ng: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

// cleanIPA normalises espeak output: drops ties, ZWJ and language-switch
// markers, collapses whitespace.
func cleanIPA(s string) string {
	s = strings.ReplaceAll(s, "͡", "")
	s = strings.ReplaceAll(s, "͜", "")
	s = strings.ReplaceAll(s, "‍", "")
	s = langSwitchRe.ReplaceAllString(s, "")
	return strings.TrimSpace(spaceRe.ReplaceAllString(s, " "))
}

// Phonemize returns IPA for text with punctuation preserved (words separated
// by spaces, punctuation attached to the preceding word) — the same shape
// the phonemizer library produces with preserve_punctuation=True.
func (p *Phonemizer) Phonemize(text string) (string, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return "", nil
	}
	parts := punctSplitRe.Split(text, -1)
	puncts := punctSplitRe.FindAllString(text, -1)

	var lines []string
	var lineIdx []int
	for i, part := range parts {
		if strings.TrimSpace(part) != "" {
			lines = append(lines, strings.TrimSpace(part))
			lineIdx = append(lineIdx, i)
		}
	}
	phones := make([]string, len(parts))
	if len(lines) > 0 {
		out, err := p.Run(p.Voice, lines)
		if err != nil {
			return "", err
		}
		for j, idx := range lineIdx {
			if j < len(out) {
				phones[idx] = cleanIPA(out[j])
			}
		}
	}

	var sb strings.Builder
	for i, ph := range phones {
		if ph != "" {
			if sb.Len() > 0 && !strings.HasSuffix(sb.String(), " ") {
				sb.WriteString(" ")
			}
			sb.WriteString(ph)
		}
		if i < len(puncts) {
			sb.WriteString(puncts[i])
		}
	}
	return strings.TrimSpace(sb.String()), nil
}
