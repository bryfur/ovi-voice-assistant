// Package tts turns text into speech on sherpa-onnx (Kokoro, Piper) and
// streams a model's reply into audio sentence by sentence.
package tts

import (
	"fmt"
	"log/slog"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/models"
)

// Synthesizer renders text to 16-bit mono PCM.
type Synthesizer interface {
	Load() error
	SampleRate() int
	// Synthesize renders text, calling emit with PCM as each sentence is ready.
	Synthesize(text string, emit func(pcm []byte) error) error
	Close()
}

const (
	kokoroPack   = "kokoro-multi-lang-v1_0" // fp32: the int8 pack is ~3x slower on x86
	defaultVoice = "af_heart"
	defaultPiper = "en_US-lessac-medium"
)

// kokoroVoices maps Kokoro v1.0 voice names to speaker ids in voices.bin.
var kokoroVoices = func() map[string]int {
	names := strings.Fields(`af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova
		af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck
		am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis ef_dora
		em_alex ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola jf_alpha jf_gongitsune
		jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa zf_xiaobei zf_xiaoni zf_xiaoxiao
		zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang`)
	m := make(map[string]int, len(names))
	for i, n := range names {
		m[n] = i
	}
	return m
}()

// KokoroVoices lists the English Kokoro voices, sorted.
func KokoroVoices() []string {
	var out []string
	for name := range kokoroVoices {
		if name[0] == 'a' || name[0] == 'b' {
			out = append(out, name)
		}
	}
	slices.Sort(out)
	return out
}

// New creates the configured synthesizer, outputting PCM at rate (0 = the
// model's native rate).
func New(cfg config.TTSConfig, rate int) (Synthesizer, error) {
	switch cfg.Provider {
	case "kokoro", "piper":
		return &synth{cfg: cfg, rate: rate}, nil
	}
	return nil, fmt.Errorf("unknown TTS provider %q", cfg.Provider)
}

// synth is Kokoro or Piper through sherpa-onnx's offline TTS.
type synth struct {
	cfg  config.TTSConfig
	rate int
	sid  int // speaker id

	mu  sync.Mutex
	tts *sherpa.OfflineTts
}

func (s *synth) SampleRate() int { return s.rate }

func (s *synth) Load() error {
	mc := sherpa.OfflineTtsModelConfig{NumThreads: min(runtime.NumCPU(), 8), Provider: "cpu"}
	var err error
	if s.cfg.Provider == "kokoro" {
		mc.Kokoro, err = s.kokoro()
	} else {
		mc.Vits, err = s.piper()
	}
	if err != nil {
		return err
	}
	cfg := sherpa.OfflineTtsConfig{Model: mc, MaxNumSentences: 1}
	if s.tts = sherpa.NewOfflineTts(&cfg); s.tts == nil {
		return fmt.Errorf("sherpa-onnx failed to load %s TTS", s.cfg.Provider)
	}
	native := s.tts.SampleRate()
	if s.rate == 0 {
		s.rate = native
	}
	slog.Info("TTS ready", "provider", s.cfg.Provider, "voice", s.cfg.Model, "native_hz", native, "output_hz", s.rate)
	return nil
}

func (s *synth) kokoro() (c sherpa.OfflineTtsKokoroModelConfig, err error) {
	voice := s.cfg.Model
	if voice == "" {
		voice = defaultVoice
	}
	if id, ok := kokoroVoices[voice]; ok {
		s.sid = id
	} else if s.sid, err = strconv.Atoi(voice); err != nil {
		return c, fmt.Errorf("unknown Kokoro voice %q", voice)
	}
	dir, err := models.Ensure(models.TTS, kokoroPack)
	if err != nil {
		return c, err
	}
	model, err := models.Find(dir, "model.onnx", "model.int8.onnx")
	if err != nil {
		return c, err
	}
	lexicon := "lexicon-us-en.txt"
	if strings.HasPrefix(voice, "b") {
		lexicon = "lexicon-gb-en.txt"
	}
	return sherpa.OfflineTtsKokoroModelConfig{
		Model:       model,
		Voices:      filepath.Join(dir, "voices.bin"),
		Tokens:      filepath.Join(dir, "tokens.txt"),
		DataDir:     filepath.Join(dir, "espeak-ng-data"),
		Lexicon:     filepath.Join(dir, lexicon),
		LengthScale: 1,
	}, nil
}

func (s *synth) piper() (c sherpa.OfflineTtsVitsModelConfig, err error) {
	name := s.cfg.Model
	if name == "" {
		name = defaultPiper
	}
	dir, err := models.Ensure(models.TTS, "vits-piper-"+name)
	if err != nil {
		return c, err
	}
	model, err := models.Find(dir, "*.onnx")
	if err != nil {
		return c, err
	}
	return sherpa.OfflineTtsVitsModelConfig{
		Model:       model,
		Tokens:      filepath.Join(dir, "tokens.txt"),
		DataDir:     filepath.Join(dir, "espeak-ng-data"),
		NoiseScale:  0.667,
		NoiseScaleW: 0.8,
		LengthScale: 1,
	}, nil
}

func (s *synth) Synthesize(text string, emit func([]byte) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tts == nil {
		return fmt.Errorf("TTS not loaded")
	}
	speed := float32(1)
	if s.cfg.Speed > 0 {
		speed = float32(s.cfg.Speed)
	}
	native := s.tts.SampleRate()
	var err error
	s.tts.GenerateWithCallback(text, s.sid, speed, func(samples []float32) bool {
		if len(samples) > 0 {
			err = emit(pcm(resample(samples, native, s.rate)))
		}
		return err == nil
	})
	return err
}

func (s *synth) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tts != nil {
		sherpa.DeleteOfflineTts(s.tts)
		s.tts = nil
	}
}
