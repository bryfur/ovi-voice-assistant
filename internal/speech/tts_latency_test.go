package speech

import (
	"os"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Time-to-first-audio and real-time factor for Kokoro; OVI_TEST_MODELS=1.
func TestKokoroLatency(t *testing.T) {
	if os.Getenv("OVI_TEST_MODELS") != "1" {
		t.Skip("set OVI_TEST_MODELS=1")
	}
	s, _ := NewTTS(config.TTSConfig{Provider: "kokoro", Model: "af_heart", Speed: 1}, 24000)
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
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
		t.Logf("%-3d words: first audio after %v, total %v, audio %v, RTF %.2f",
			len(splitWords(text)), first.Round(time.Millisecond), total.Round(time.Millisecond), audio.Round(time.Millisecond), total.Seconds()/audio.Seconds())
	}
}

func splitWords(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ' ' {
			if cur != "" {
				out = append(out, cur)
			}
			cur = ""
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}
