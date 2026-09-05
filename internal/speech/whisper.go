package speech

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

const whisperDefault = "base.en"

// whisper runs OpenAI Whisper offline: the VAD collects one utterance,
// then the whole segment is decoded.
type whisper struct {
	cfg config.STTConfig
	mu  sync.Mutex
	vad *sileroVAD
	rec *sherpa.OfflineRecognizer
}

func newWhisper(cfg config.STTConfig) *whisper { return &whisper{cfg: cfg} }

func (w *whisper) Load() error {
	name := w.cfg.Model
	if name == "" {
		name = whisperDefault
	}
	dir, err := ensurePack(asrRelease, "sherpa-onnx-whisper-"+name)
	if err != nil {
		return err
	}
	enc, err := findFile(dir, "*-encoder.int8.onnx", "*-encoder.onnx")
	if err != nil {
		return err
	}
	dec, err := findFile(dir, "*-decoder.int8.onnx", "*-decoder.onnx")
	if err != nil {
		return err
	}
	tokens, err := findFile(dir, "*-tokens.txt", "tokens.txt")
	if err != nil {
		return err
	}
	lang := w.cfg.Language
	if strings.HasSuffix(name, ".en") {
		lang = "en"
	}
	rc := sherpa.OfflineRecognizerConfig{DecodingMethod: "greedy_search"}
	rc.FeatConfig = sherpa.FeatureConfig{SampleRate: micRate, FeatureDim: 80}
	rc.ModelConfig = sherpa.OfflineModelConfig{
		Tokens: tokens, NumThreads: threads(), Provider: "cpu", ModelType: "whisper",
	}
	rc.ModelConfig.Whisper = sherpa.OfflineWhisperModelConfig{
		Encoder: enc, Decoder: dec, Language: lang, Task: "transcribe", TailPaddings: -1,
	}
	if w.rec = sherpa.NewOfflineRecognizer(&rc); w.rec == nil {
		return errLoad("Whisper " + name)
	}
	if w.vad, err = newSileroVAD(w.cfg.Silence); err != nil {
		return err
	}
	slog.Info("Whisper STT ready", "model", name)
	return nil
}

func (w *whisper) Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	seg, err := listen(ctx, mic, w.vad, onSpeech, nil)
	if err != nil || len(seg) < micRate/10 {
		return "", err
	}
	stream := sherpa.NewOfflineStream(w.rec)
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(micRate, seg)
	w.rec.Decode(stream)
	return strings.TrimSpace(stream.GetResult().Text), nil
}

func (w *whisper) Close() {
	if w.rec != nil {
		sherpa.DeleteOfflineRecognizer(w.rec)
	}
	if w.vad != nil {
		w.vad.Close()
	}
}
