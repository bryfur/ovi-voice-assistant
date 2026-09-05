package tts

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	onnx "github.com/yalue/onnxruntime_go"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/download"
	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
	"github.com/bryfur/ovi-voice-assistant/internal/ort"
)

const piperVoicesRepo = "rhasspy/piper-voices"

// PiperConfig is the subset of a voice's .onnx.json we use.
type PiperConfig struct {
	Audio struct {
		SampleRate int `json:"sample_rate"`
	} `json:"audio"`
	Espeak struct {
		Voice string `json:"voice"`
	} `json:"espeak"`
	Inference struct {
		NoiseScale  float32 `json:"noise_scale"`
		LengthScale float32 `json:"length_scale"`
		NoiseW      float32 `json:"noise_w"`
	} `json:"inference"`
	PhonemeType  string             `json:"phoneme_type"`
	PhonemeIDMap map[string][]int64 `json:"phoneme_id_map"`
	NumSpeakers  int                `json:"num_speakers"`
	SpeakerIDMap map[string]int     `json:"speaker_id_map"`
}

// PiperPhonemesToIDs maps phonemes to ids: BOS, then each phoneme
// followed by PAD, then EOS. Unknown phonemes are skipped.
func PiperPhonemesToIDs(phonemes string, idMap map[string][]int64) []int64 {
	var ids []int64
	ids = append(ids, idMap["^"]...)
	for _, r := range phonemes {
		p, ok := idMap[string(r)]
		if !ok {
			continue
		}
		ids = append(ids, p...)
		ids = append(ids, idMap["_"]...)
	}
	return append(ids, idMap["$"]...)
}

var piperVoiceNameRe = regexp.MustCompile(`^([a-z]{2,3})_([A-Za-z]{2,})-([^-]+)-(x_low|low|medium|high)$`)

// PiperVoiceFiles returns the HF repo paths for a voice like en_US-lessac-medium.
func PiperVoiceFiles(name string) (model, cfg string, err error) {
	m := piperVoiceNameRe.FindStringSubmatch(name)
	if m == nil {
		return "", "", fmt.Errorf("unrecognised piper voice name %q (expected e.g. en_US-lessac-medium)", name)
	}
	lang := m[1]
	code := m[1] + "_" + m[2]
	base := lang + "/" + code + "/" + m[3] + "/" + m[4] + "/" + name
	return base + ".onnx", base + ".onnx.json", nil
}

// PiperTTS is text-to-speech using Piper voices with sentence-level streaming.
type PiperTTS struct {
	settings   *config.Settings
	targetRate int // 0 = native model rate
	sampleRate int

	mu         sync.Mutex
	session    *ort.Session
	cfg        PiperConfig
	phonemizer *Phonemizer

	// ModelsDir holds cached voices.
	ModelsDir string
}

// NewPiperTTS creates an unloaded provider targeting sampleRate (0 = native).
func NewPiperTTS(settings *config.Settings, sampleRate int) *PiperTTS {
	return &PiperTTS{
		settings:   settings,
		targetRate: sampleRate,
		sampleRate: sampleRate,
		ModelsDir:  filepath.Join(config.CacheDir(), "piper"),
	}
}

func (p *PiperTTS) SampleRate() int  { return p.sampleRate }
func (p *PiperTTS) SampleWidth() int { return 2 }
func (p *PiperTTS) Channels() int    { return 1 }

func (p *PiperTTS) resolveModel(name string) (string, string, error) {
	if st, err := os.Stat(name); err == nil && !st.IsDir() {
		return name, name + ".json", nil
	}
	modelRel, cfgRel, err := PiperVoiceFiles(name)
	if err != nil {
		return "", "", err
	}
	modelPath, err := download.Ensure(filepath.Join(p.ModelsDir, name+".onnx"), download.HuggingFaceURL(piperVoicesRepo, modelRel))
	if err != nil {
		return "", "", err
	}
	cfgPath, err := download.Ensure(filepath.Join(p.ModelsDir, name+".onnx.json"), download.HuggingFaceURL(piperVoicesRepo, cfgRel))
	if err != nil {
		return "", "", err
	}
	return modelPath, cfgPath, nil
}

// Load downloads the voice if needed and creates the session.
func (p *PiperTTS) Load() error {
	if err := CheckEspeak(); err != nil {
		return err
	}
	modelPath, cfgPath, err := p.resolveModel(p.settings.TTS.Model)
	if err != nil {
		return err
	}
	slog.Info("Loading piper TTS model", "path", modelPath)
	cfgData, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	var cfg PiperConfig
	if err := json.Unmarshal(cfgData, &cfg); err != nil {
		return fmt.Errorf("parse %s: %w", cfgPath, err)
	}
	if cfg.Audio.SampleRate == 0 {
		cfg.Audio.SampleRate = 22050
	}
	if cfg.Espeak.Voice == "" {
		cfg.Espeak.Voice = "en-us"
	}
	if cfg.Inference.LengthScale == 0 {
		cfg.Inference.LengthScale = 1
	}
	if cfg.Inference.NoiseScale == 0 {
		cfg.Inference.NoiseScale = 0.667
	}
	if cfg.Inference.NoiseW == 0 {
		cfg.Inference.NoiseW = 0.8
	}
	sess, err := ort.NewSession(modelPath, ort.SessionConfig{IntraOpThreads: 4, InterOpThreads: 1})
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.session = sess
	p.cfg = cfg
	p.phonemizer = NewPhonemizer(cfg.Espeak.Voice)
	if p.targetRate == 0 {
		p.sampleRate = cfg.Audio.SampleRate
	}
	p.mu.Unlock()
	slog.Info("TTS model loaded", "native_hz", cfg.Audio.SampleRate, "output_hz", p.sampleRate)
	return nil
}

// Close frees the session.
func (p *PiperTTS) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.session != nil {
		p.session.Destroy()
		p.session = nil
	}
}

var piperSentenceRe = regexp.MustCompile(`[^.!?]+[.!?]*`)

// synthesizeSentence runs one sentence through the model and returns PCM at
// the native rate.
func (p *PiperTTS) synthesizeSentence(sentence string) ([]int16, error) {
	phonemes, err := p.phonemizer.Phonemize(sentence)
	if err != nil {
		return nil, err
	}
	ids := PiperPhonemesToIDs(phonemes, p.cfg.PhonemeIDMap)
	if len(ids) == 0 {
		return nil, nil
	}
	lengthScale := p.cfg.Inference.LengthScale
	if p.settings.TTS.LengthScale > 0 {
		lengthScale = float32(p.settings.TTS.LengthScale)
	}
	inputT, err := p.session.IntTensorFor("input", []int64{1, int64(len(ids))}, ids)
	if err != nil {
		return nil, err
	}
	lenT, err := p.session.IntTensorFor("input_lengths", []int64{1}, []int64{int64(len(ids))})
	if err != nil {
		inputT.Destroy()
		return nil, err
	}
	scalesT, err := ort.FloatTensor([]int64{3}, []float32{p.cfg.Inference.NoiseScale, lengthScale, p.cfg.Inference.NoiseW})
	if err != nil {
		inputT.Destroy()
		lenT.Destroy()
		return nil, err
	}
	inputs := map[string]onnx.Value{"input": inputT, "input_lengths": lenT, "scales": scalesT}
	toDestroy := []onnx.Value{inputT, lenT, scalesT}
	if p.session.HasInput("sid") {
		sid := int64(0)
		if p.settings.TTS.SpeakerID != nil {
			sid = int64(*p.settings.TTS.SpeakerID)
		}
		sidT, err := p.session.IntTensorFor("sid", []int64{1}, []int64{sid})
		if err != nil {
			ort.DestroyAll(toDestroy)
			return nil, err
		}
		inputs["sid"] = sidT
		toDestroy = append(toDestroy, sidT)
	}
	outputs, err := p.session.RunNamed(inputs)
	ort.DestroyAll(toDestroy)
	if err != nil {
		return nil, err
	}
	defer ort.DestroyAll(outputs)
	samples, _, err := ort.FloatData(outputs[0])
	if err != nil {
		return nil, err
	}
	// Normalise to peak like piper's audio_float_to_int16.
	var peak float32 = 0.01
	for _, s := range samples {
		if a := float32(math.Abs(float64(s))); a > peak {
			peak = a
		}
	}
	return dsp.Float32ToInt16(samples, 32767/peak), nil
}

// SynthesizeIter renders each sentence and emits it as soon as it is ready.
func (p *PiperTTS) SynthesizeIter(text string, emit func(pcm []byte) error) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.session == nil {
		return fmt.Errorf("call Load() first")
	}
	silence := int(p.settings.TTS.SentenceSilence * float64(p.cfg.Audio.SampleRate))
	for _, sentence := range piperSentenceRe.FindAllString(text, -1) {
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}
		pcm, err := p.synthesizeSentence(sentence)
		if err != nil {
			return err
		}
		if len(pcm) == 0 {
			continue
		}
		if silence > 0 {
			pcm = append(pcm, make([]int16, silence)...)
		}
		if p.sampleRate != p.cfg.Audio.SampleRate {
			pcm = dsp.Resample(pcm, p.cfg.Audio.SampleRate, p.sampleRate)
		}
		if err := emit(dsp.Int16ToBytes(pcm)); err != nil {
			return err
		}
	}
	return nil
}

// Synthesize renders text to PCM at SampleRate.
func (p *PiperTTS) Synthesize(text string) ([]byte, error) {
	return SynthesizeAll(p, text)
}
