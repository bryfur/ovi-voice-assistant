package tts

import (
	"os"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Piper time-to-first-audio and RTF; OVI_TEST_PIPER=1 (downloads ~70 MB).
func TestPiperLatency(t *testing.T) {
	if os.Getenv("OVI_TEST_PIPER") != "1" {
		t.Skip("set OVI_TEST_PIPER=1")
	}
	s, _ := New(config.TTSConfig{Provider: "piper", Model: "en_US-lessac-medium", Speed: 1}, 24000)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Synthesize("Warm up.", func([]byte) error { return nil })
	for _, text := range []string{
		"Sure.",
		"The weather today is sunny with a high of seventy two degrees.",
	} {
		start := time.Now()
		var bytes int
		s.Synthesize(text, func(pcm []byte) error { bytes += len(pcm); return nil })
		took := time.Since(start)
		audio := time.Duration(float64(bytes) / 2 / 24000 * float64(time.Second))
		t.Logf("%q: %v for %v audio, RTF %.2f", text[:min(20, len(text))], took.Round(time.Millisecond), audio.Round(time.Millisecond), took.Seconds()/audio.Seconds())
	}
}
