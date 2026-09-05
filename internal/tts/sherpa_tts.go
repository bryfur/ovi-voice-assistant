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
	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
	"github.com/bryfur/ovi-voice-assistant/internal/models"
)

const (
	kokoroPack   = "kokoro-int8-multi-lang-v1_0"
	defaultVoice = "af_heart"
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

// Sherpa is Kokoro or Piper via sherpa-onnx's offline TTS.
type Sherpa struct {
	cfg   config.TTSConfig
	rate  int // output rate
	mu    sync.Mutex
	tts   *sherpa.OfflineTts
	sid   int
	speed float32
}

// New creates a provider that outputs PCM at rate (0 = model native).
func New(cfg config.TTSConfig, rate int) (*Sherpa, error) {
	switch cfg.Provider {
	case "kokoro", "piper":
	default:
		return nil, fmt.Errorf("unknown TTS provider %q", cfg.Provider)
	}
	return &Sherpa{cfg: cfg, rate: rate, speed: 1}, nil
}

func (s *Sherpa) SampleRate() int { return s.rate }

func (s *Sherpa) Load() error {
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
	if s.cfg.Speed > 0 {
		s.speed = float32(s.cfg.Speed)
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

func (s *Sherpa) kokoro() (sherpa.OfflineTtsKokoroModelConfig, error) {
	var c sherpa.OfflineTtsKokoroModelConfig
	voice := s.cfg.Model
	if voice == "" {
		voice = defaultVoice
	}
	if id, ok := kokoroVoices[voice]; ok {
		s.sid = id
	} else if id, err := strconv.Atoi(voice); err == nil {
		s.sid = id
	} else {
		return c, fmt.Errorf("unknown Kokoro voice %q", voice)
	}
	dir, err := models.Ensure(models.TTS, kokoroPack)
	if err != nil {
		return c, err
	}
	model, err := models.Find(dir, "model.int8.onnx", "model.onnx")
	if err != nil {
		return c, err
	}
	lexicon := "lexicon-us-en.txt"
	if strings.HasPrefix(voice, "b") {
		lexicon = "lexicon-gb-en.txt"
	}
	c = sherpa.OfflineTtsKokoroModelConfig{
		Model:       model,
		Voices:      filepath.Join(dir, "voices.bin"),
		Tokens:      filepath.Join(dir, "tokens.txt"),
		DataDir:     filepath.Join(dir, "espeak-ng-data"),
		Lexicon:     filepath.Join(dir, lexicon),
		LengthScale: 1,
	}
	return c, nil
}

func (s *Sherpa) piper() (sherpa.OfflineTtsVitsModelConfig, error) {
	var c sherpa.OfflineTtsVitsModelConfig
	name := s.cfg.Model
	if name == "" {
		name = "en_US-lessac-medium"
	}
	dir, err := models.Ensure(models.TTS, "vits-piper-"+name)
	if err != nil {
		return c, err
	}
	model, err := models.Find(dir, "*.onnx")
	if err != nil {
		return c, err
	}
	c = sherpa.OfflineTtsVitsModelConfig{
		Model:       model,
		Tokens:      filepath.Join(dir, "tokens.txt"),
		DataDir:     filepath.Join(dir, "espeak-ng-data"),
		NoiseScale:  0.667,
		NoiseScaleW: 0.8,
		LengthScale: 1,
	}
	return c, nil
}

// Synthesize renders text sentence by sentence, emitting each as PCM.
func (s *Sherpa) Synthesize(text string, emit func([]byte) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tts == nil {
		return fmt.Errorf("TTS not loaded")
	}
	native := s.tts.SampleRate()
	var emitErr error
	s.tts.GenerateWithCallback(text, s.sid, s.speed, func(samples []float32) bool {
		if len(samples) == 0 {
			return true
		}
		pcm := dsp.Float32ToBytes(dsp.Resample(samples, native, s.rate))
		emitErr = emit(pcm)
		return emitErr == nil
	})
	return emitErr
}

func (s *Sherpa) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tts != nil {
		sherpa.DeleteOfflineTts(s.tts)
		s.tts = nil
	}
}
