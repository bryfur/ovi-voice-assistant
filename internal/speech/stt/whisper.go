package stt

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/models"
)

const whisperDefault = "base.en"

// whisper runs OpenAI Whisper offline: the VAD collects one utterance,
// then the whole of it is decoded.
type whisper struct {
	cfg config.STTConfig
	mu  sync.Mutex
	vad *silero
	rec *sherpa.OfflineRecognizer
}

func (w *whisper) Load() error {
	name := w.cfg.Model
	if name == "" {
		name = whisperDefault
	}
	dir, err := models.Ensure(models.ASR, "sherpa-onnx-whisper-"+name)
	if err != nil {
		return err
	}
	var enc, dec, tokens string
	for _, f := range []struct {
		dst   *string
		globs []string
	}{
		{&enc, []string{"*-encoder.int8.onnx", "*-encoder.onnx"}},
		{&dec, []string{"*-decoder.int8.onnx", "*-decoder.onnx"}},
		{&tokens, []string{"*-tokens.txt", "tokens.txt"}},
	} {
		if *f.dst, err = models.Find(dir, f.globs...); err != nil {
			return err
		}
	}
	lang := w.cfg.Language
	if strings.HasSuffix(name, ".en") {
		lang = "en"
	}
	rc := sherpa.OfflineRecognizerConfig{DecodingMethod: "greedy_search"}
	rc.FeatConfig = sherpa.FeatureConfig{SampleRate: SampleRate, FeatureDim: 80}
	rc.ModelConfig = sherpa.OfflineModelConfig{Tokens: tokens, NumThreads: threads(), Provider: "cpu", ModelType: "whisper"}
	rc.ModelConfig.Whisper = sherpa.OfflineWhisperModelConfig{Encoder: enc, Decoder: dec, Language: lang, Task: "transcribe", TailPaddings: -1}
	if w.rec = sherpa.NewOfflineRecognizer(&rc); w.rec == nil {
		return errLoad("Whisper " + name)
	}
	if w.vad, err = newSilero(w.cfg.Silence); err != nil {
		return err
	}
	slog.Info("Whisper STT ready", "model", name)
	return nil
}

func (w *whisper) Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	seg, err := listen(ctx, mic, w.vad, onSpeech, nil)
	if err != nil || len(seg) < SampleRate/10 {
		return "", err
	}
	stream := sherpa.NewOfflineStream(w.rec)
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(SampleRate, seg)
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
