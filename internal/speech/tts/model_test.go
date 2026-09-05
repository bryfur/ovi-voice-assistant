package tts

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func realSynth(t *testing.T, provider, voice, gate string) Synthesizer {
	t.Helper()
	if os.Getenv(gate) != "1" {
		t.Skipf("set %s=1", gate)
	}
	s, _ := New(config.TTSConfig{Provider: provider, Model: voice, Speed: 1}, 24000)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

// Real Kokoro synthesis through sherpa-onnx (downloads ~130 MB); OVI_TEST_MODELS=1.
func TestKokoroRealSynthesis(t *testing.T) {
	s := realSynth(t, "kokoro", "af_heart", "OVI_TEST_MODELS")
	var chunks, bytes int

	err := s.Synthesize("Hello there. This is Ovi speaking.", func(pcm []byte) error {
		chunks++
		bytes += len(pcm)
		return nil
	})

	secs := float64(bytes) / 2 / 24000
	t.Logf("chunks=%d seconds=%.2f rate=%d", chunks, secs, s.SampleRate())
	if err != nil || chunks < 2 || secs < 1 || secs > 6 {
		t.Fatalf("err=%v chunks=%d secs=%.2f", err, chunks, secs)
	}
}

// Time-to-first-audio and real-time factor; OVI_TEST_MODELS=1 (Kokoro),
// OVI_TEST_PIPER=1 (Piper, downloads ~70 MB).
func TestLatency(t *testing.T) {
	for _, m := range []struct{ provider, voice, gate string }{
		{"kokoro", "af_heart", "OVI_TEST_MODELS"}, {"piper", "en_US-lessac-medium", "OVI_TEST_PIPER"},
	} {
		t.Run(m.provider, func(t *testing.T) {
			s := realSynth(t, m.provider, m.voice, m.gate)
			s.Synthesize("Warm up.", func([]byte) error { return nil })
			for _, text := range []string{
				"Sure.",
				"The weather today is sunny with a high of seventy two degrees.",
				"The weather today is sunny with a high of seventy two degrees, and there is a slight chance of rain in the evening, so you may want to bring an umbrella.",
			} {
				start := time.Now()
				var first time.Duration
				var bytes int
				s.Synthesize(text, func(pcm []byte) error {
					if first == 0 {
						first = time.Since(start)
					}
					bytes += len(pcm)
					return nil
				})
				total := time.Since(start)
				audio := time.Duration(float64(bytes) / 2 / 24000 * float64(time.Second))
				t.Logf("%-3d words: first audio after %v, total %v, audio %v, RTF %.2f", len(strings.Fields(text)),
					first.Round(time.Millisecond), total.Round(time.Millisecond), audio.Round(time.Millisecond), total.Seconds()/audio.Seconds())
			}
		})
	}
}
