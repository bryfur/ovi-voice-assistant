package stt

import (
	"context"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/models"
	"log/slog"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

const (
	nemotronPack    = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-%s-int8-2026-04-25"
	nemotronDefault = "560ms"
)

var nemotronChunks = map[string]bool{"80ms": true, "160ms": true, "560ms": true, "1120ms": true}

// nemotron is NVIDIA Nemotron Speech 600M, a cache-aware streaming
// transducer: audio is decoded while the user is still talking, so the
// transcript is ready the moment the VAD closes the segment.
type nemotron struct {
	cfg config.STTConfig
	mu  sync.Mutex
	vad *sileroVAD
	rec *sherpa.OnlineRecognizer
}

func newNemotron(cfg config.STTConfig) *nemotron { return &nemotron{cfg: cfg} }

func (n *nemotron) Load() error {
	chunk := n.cfg.Model
	if !nemotronChunks[chunk] {
		chunk = nemotronDefault
	}
	dir, err := models.Ensure(models.ASR, fmt.Sprintf(nemotronPack, chunk))
	if err != nil {
		return err
	}
	enc, err := models.Find(dir, "encoder*.int8.onnx", "encoder*.onnx")
	if err != nil {
		return err
	}
	dec, err := models.Find(dir, "decoder*.int8.onnx", "decoder*.onnx")
	if err != nil {
		return err
	}
	join, err := models.Find(dir, "joiner*.int8.onnx", "joiner*.onnx")
	if err != nil {
		return err
	}
	tokens, err := models.Find(dir, "tokens.txt")
	if err != nil {
		return err
	}
	rc := sherpa.OnlineRecognizerConfig{DecodingMethod: "greedy_search"}
	rc.FeatConfig = sherpa.FeatureConfig{SampleRate: SampleRate, FeatureDim: 128}
	rc.ModelConfig = sherpa.OnlineModelConfig{
		Tokens: tokens, NumThreads: threads(), Provider: "cpu", ModelType: "nemotron",
	}
	rc.ModelConfig.Transducer = sherpa.OnlineTransducerModelConfig{Encoder: enc, Decoder: dec, Joiner: join}
	if n.rec = sherpa.NewOnlineRecognizer(&rc); n.rec == nil {
		return errLoad("Nemotron")
	}
	if n.vad, err = newSileroVAD(n.cfg.Silence); err != nil {
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
	seg, err := listen(ctx, mic, n.vad, onSpeech, func(samples []float32) {
		stream.AcceptWaveform(SampleRate, samples)
		for n.rec.IsReady(stream) {
			n.rec.Decode(stream)
		}
	})
	if err != nil || seg == nil {
		return "", err
	}
	stream.InputFinished()
	for n.rec.IsReady(stream) {
		n.rec.Decode(stream)
	}
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
