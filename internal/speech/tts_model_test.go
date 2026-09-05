package speech

import (
	"os"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Real Kokoro synthesis through sherpa-onnx (downloads ~130 MB on first
// run); enabled with OVI_TEST_MODELS=1.
func TestKokoroRealSynthesis(t *testing.T) {
	if os.Getenv("OVI_TEST_MODELS") != "1" {
		t.Skip("set OVI_TEST_MODELS=1")
	}
	s, err := NewTTS(config.TTSConfig{Provider: "kokoro", Model: "af_heart", Speed: 1}, 24000)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Load(); err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var chunks, bytes int

	err = s.Synthesize("Hello there. This is Ovi speaking.", func(pcm []byte) error {
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
