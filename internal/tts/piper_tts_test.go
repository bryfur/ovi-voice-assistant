package tts

import (
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func TestPiperPhonemesToIDs(t *testing.T) {
	idMap := map[string][]int64{"^": {1}, "$": {2}, "_": {0}, "h": {20}, "i": {21}, " ": {3}}

	ids := PiperPhonemesToIDs("hi ?", idMap)

	want := []int64{1, 20, 0, 21, 0, 3, 0, 2}
	if len(ids) != len(want) {
		t.Fatalf("got %v", ids)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("got %v want %v", ids, want)
		}
	}
}

func TestPiperVoiceFiles(t *testing.T) {
	model, cfg, err := PiperVoiceFiles("en_US-lessac-medium")

	if err != nil || model != "en/en_US/lessac/medium/en_US-lessac-medium.onnx" || cfg != model+".json" {
		t.Fatalf("got %q %q %v", model, cfg, err)
	}
	if _, _, err := PiperVoiceFiles("bogus"); err == nil {
		t.Fatal("expected error")
	}
}

func TestPiperSynthesizeBeforeLoad(t *testing.T) {
	p := NewPiperTTS(config.Default(), 0)

	_, err := p.Synthesize("hi")

	if err == nil || p.SampleRate() != 0 {
		t.Fatalf("err=%v rate=%d", err, p.SampleRate())
	}
}
