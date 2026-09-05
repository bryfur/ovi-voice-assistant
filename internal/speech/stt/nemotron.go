package stt

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/models"
)

const (
	nemotronPack    = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-%s-int8-2026-04-25"
	nemotronDefault = "560ms"
)

var nemotronChunks = map[string]bool{"80ms": true, "160ms": true, "560ms": true, "1120ms": true}

// nemotron is NVIDIA Nemotron Speech 600M, a cache-aware streaming
// transducer: audio is decoded while the user is still talking, so the
// transcript is ready the moment the VAD closes the utterance.
type nemotron struct {
	cfg config.STTConfig
	mu  sync.Mutex
	vad *silero
	rec *sherpa.OnlineRecognizer
}

func (n *nemotron) Load() error {
	chunk := n.cfg.Model
	if !nemotronChunks[chunk] {
		chunk = nemotronDefault
	}
	dir, err := models.Ensure(models.ASR, fmt.Sprintf(nemotronPack, chunk))
	if err != nil {
		return err
	}
	var enc, dec, join, tokens string
	for _, f := range []struct {
		dst   *string
		globs []string
	}{
		{&enc, []string{"encoder*.int8.onnx", "encoder*.onnx"}},
		{&dec, []string{"decoder*.int8.onnx", "decoder*.onnx"}},
		{&join, []string{"joiner*.int8.onnx", "joiner*.onnx"}},
		{&tokens, []string{"tokens.txt"}},
	} {
		if *f.dst, err = models.Find(dir, f.globs...); err != nil {
			return err
		}
	}
	rc := sherpa.OnlineRecognizerConfig{DecodingMethod: "greedy_search"}
	rc.FeatConfig = sherpa.FeatureConfig{SampleRate: SampleRate, FeatureDim: 128}
	rc.ModelConfig = sherpa.OnlineModelConfig{Tokens: tokens, NumThreads: threads(), Provider: "cpu", ModelType: "nemotron"}
	rc.ModelConfig.Transducer = sherpa.OnlineTransducerModelConfig{Encoder: enc, Decoder: dec, Joiner: join}
	if n.rec = sherpa.NewOnlineRecognizer(&rc); n.rec == nil {
		return errLoad("Nemotron")
	}
	if n.vad, err = newSilero(n.cfg.Silence); err != nil {
		return err
	}
	slog.Info("Nemotron STT ready", "chunk", chunk)
	return nil
}

func (n *nemotron) Listen(ctx context.Context, mic <-chan []byte, onSpeech func()) (string, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	stream := sherpa.NewOnlineStream(n.rec)
	defer sherpa.DeleteOnlineStream(stream)
	decode := func() {
		for n.rec.IsReady(stream) {
			n.rec.Decode(stream)
		}
	}
	seg, err := listen(ctx, mic, n.vad, onSpeech, func(s []float32) {
		stream.AcceptWaveform(SampleRate, s)
		decode()
	})
	if err != nil || seg == nil {
		return "", err
	}
	stream.InputFinished()
	decode()
	return n.rec.GetResult(stream).Text, nil
}

func (n *nemotron) Close() {
	if n.rec != nil {
		sherpa.DeleteOnlineRecognizer(n.rec)
	}
	if n.vad != nil {
		n.vad.Close()
	}
}
