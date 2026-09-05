package tts

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"

	onnx "github.com/yalue/onnxruntime_go"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/download"
	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
	"github.com/bryfur/ovi-voice-assistant/internal/ort"
)

const (
	kokoroModelURL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX" +
		"/resolve/main/onnx/model_uint8.onnx"
	kokoroVoicesURL = "https://github.com/thewh1teagle/kokoro-onnx" +
		"/releases/download/model-files-v1.0/voices-v1.0.bin"
	kokoroNativeRate   = 24000
	kokoroMaxPhonemes  = 510
	kokoroStyleDim     = 256
	kokoroDefaultVoice = "af_heart"
)

// kokoroSymbols is the Kokoro v1.0 symbol table; index = token id.
var kokoroSymbols = func() []string {
	pad := "$"
	punctuation := ";:,.!?¡¿—…\"«»“” "
	letters := "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	ipa := "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
	var syms []string
	syms = append(syms, pad)
	for _, r := range punctuation {
		syms = append(syms, string(r))
	}
	for _, r := range letters {
		syms = append(syms, string(r))
	}
	for _, r := range ipa {
		syms = append(syms, string(r))
	}
	return syms
}()

// KokoroVocab maps a symbol to its token id.
var KokoroVocab = func() map[rune]int64 {
	m := map[rune]int64{}
	for i, s := range kokoroSymbols {
		r := []rune(s)[0]
		if _, dup := m[r]; !dup {
			m[r] = int64(i)
		}
	}
	return m
}()

// KokoroTokenize maps phonemes to token ids, skipping unknown symbols.
func KokoroTokenize(phonemes string) []int64 {
	var ids []int64
	for _, r := range phonemes {
		if id, ok := KokoroVocab[r]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

var kokoroBatchSplit = regexp.MustCompile(`([.,!?;])`)

// KokoroSplitPhonemes splits phonemes into batches under the max length,
// preferring punctuation boundaries.
func KokoroSplitPhonemes(phonemes string) []string {
	// Split by punctuation and keep the delimiters.
	var parts []string
	last := 0
	for _, loc := range kokoroBatchSplit.FindAllStringIndex(phonemes, -1) {
		parts = append(parts, phonemes[last:loc[0]], phonemes[loc[0]:loc[1]])
		last = loc[1]
	}
	parts = append(parts, phonemes[last:])

	var batches []string
	current := ""
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if len([]rune(current))+len([]rune(part))+1 >= kokoroMaxPhonemes {
			batches = append(batches, current)
			current = part
			continue
		}
		if strings.Contains(".,!?;", part) {
			current += part
		} else {
			if current != "" {
				current += " "
			}
			current += part
		}
	}
	if current != "" {
		batches = append(batches, current)
	}
	return batches
}

// kokoroPostProcess applies the kokoro-onnx phoneme fix-ups.
func kokoroPostProcess(phonemes, lang string) string {
	phonemes = strings.ReplaceAll(phonemes, "kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
	phonemes = strings.ReplaceAll(phonemes, "kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
	r := strings.NewReplacer("ʲ", "j", "r", "ɹ", "x", "k", "ɬ", "l")
	phonemes = r.Replace(phonemes)
	if lang == "en-us" {
		phonemes = strings.ReplaceAll(phonemes, "nˈaɪnti", "nˈaɪndi")
	}
	var sb strings.Builder
	for _, ch := range phonemes {
		if _, ok := KokoroVocab[ch]; ok {
			sb.WriteRune(ch)
		}
	}
	return strings.TrimSpace(sb.String())
}

var (
	kokoroQuoteRe    = strings.NewReplacer("‘", "'", "’", "'", "«", "“", "»", "”", "“", "\"", "”", "\"", "(", "«", ")", "»")
	kokoroSpaceRe    = regexp.MustCompile(`[^\S \n]`)
	kokoroMultiSpace = regexp.MustCompile(`  +`)
	kokoroAbbrevs    = []struct {
		re   *regexp.Regexp
		repl string
	}{
		// Go's regexp has no lookahead, so the following context is captured
		// and re-emitted instead.
		{regexp.MustCompile(`\bD[Rr]\.( [A-Z])`), "Doctor$1"},
		{regexp.MustCompile(`\bMr\.`), "Mister"},
		{regexp.MustCompile(`\bMR\.( [A-Z])`), "Mister$1"},
		{regexp.MustCompile(`\bMs\.`), "Miss"},
		{regexp.MustCompile(`\bMS\.( [A-Z])`), "Miss$1"},
		{regexp.MustCompile(`\bMrs\.`), "Mrs"},
		{regexp.MustCompile(`\bMRS\.( [A-Z])`), "Mrs$1"},
		{regexp.MustCompile(`\betc\.( [a-z]|$)`), "etc$1"},
		{regexp.MustCompile(`(?i)\b(y)eah?\b`), "${1}e'a"},
	}
)

// KokoroNormalizeText applies light text normalisation (quotes, spacing,
// common abbreviations) before phonemization.
func KokoroNormalizeText(text string) string {
	text = kokoroQuoteRe.Replace(text)
	text = kokoroSpaceRe.ReplaceAllString(text, " ")
	text = kokoroMultiSpace.ReplaceAllString(text, " ")
	for _, a := range kokoroAbbrevs {
		text = a.re.ReplaceAllString(text, a.repl)
	}
	return strings.TrimSpace(text)
}

// KokoroTTS is text-to-speech using Kokoro 82M (ONNX int8 quantized).
type KokoroTTS struct {
	settings   *config.Settings
	sampleRate int

	mu         sync.Mutex
	session    *ort.Session
	voices     map[string]*NPYArray
	phonemizer *Phonemizer

	// ModelsDir holds cached model files.
	ModelsDir string
}

// NewKokoroTTS creates an unloaded provider targeting sampleRate.
func NewKokoroTTS(settings *config.Settings, sampleRate int) *KokoroTTS {
	if sampleRate == 0 {
		sampleRate = kokoroNativeRate
	}
	return &KokoroTTS{
		settings:   settings,
		sampleRate: sampleRate,
		ModelsDir:  filepath.Join(config.CacheDir(), "kokoro"),
	}
}

func (k *KokoroTTS) SampleRate() int  { return k.sampleRate }
func (k *KokoroTTS) SampleWidth() int { return 2 }
func (k *KokoroTTS) Channels() int    { return 1 }

func (k *KokoroTTS) voice() string {
	if v := k.settings.TTS.Model; v != "" {
		return v
	}
	return kokoroDefaultVoice
}

func (k *KokoroTTS) lang() string {
	v := k.voice()
	if strings.HasPrefix(v, "bf_") || strings.HasPrefix(v, "bm_") {
		return "en-gb"
	}
	return "en-us"
}

// Load downloads the model + voices, creates the session and warms up.
func (k *KokoroTTS) Load() error {
	if err := CheckEspeak(); err != nil {
		return err
	}
	modelPath, err := download.Ensure(filepath.Join(k.ModelsDir, "model_uint8.onnx"), kokoroModelURL)
	if err != nil {
		return err
	}
	voicesPath, err := download.Ensure(filepath.Join(k.ModelsDir, "voices-v1.0.bin"), kokoroVoicesURL)
	if err != nil {
		return err
	}
	slog.Info("Loading Kokoro TTS model")

	providers := ort.SelectProviders("auto", true)
	sess, err := ort.NewSession(modelPath, ort.SessionConfig{
		IntraOpThreads: runtime.NumCPU(),
		InterOpThreads: 1,
		Parallel:       true,
		OptExtended:    true,
		Providers:      providers,
		CoreMLCacheDir: filepath.Join(k.ModelsDir, "coreml_cache"),
	})
	if err != nil {
		return err
	}
	vdata, err := os.ReadFile(voicesPath)
	if err != nil {
		sess.Destroy()
		return err
	}
	voices, err := ParseNPZ(vdata)
	if err != nil {
		sess.Destroy()
		return fmt.Errorf("parse voices: %w", err)
	}
	if _, ok := voices[k.voice()]; !ok {
		sess.Destroy()
		return fmt.Errorf("unknown Kokoro voice %q", k.voice())
	}
	k.mu.Lock()
	k.session = sess
	k.voices = voices
	k.phonemizer = NewPhonemizer(k.lang())
	k.mu.Unlock()
	slog.Info("Kokoro TTS loaded", "providers", sess.Providers, "native_hz", kokoroNativeRate, "output_hz", k.sampleRate)
	k.warmup()
	return nil
}

func (k *KokoroTTS) warmup() {
	if err := k.SynthesizeIter("Hello.", func([]byte) error { return nil }); err != nil {
		slog.Warn("Kokoro warmup failed", "err", err)
	}
}

// Close frees the session.
func (k *KokoroTTS) Close() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.session != nil {
		k.session.Destroy()
		k.session = nil
	}
}

// runInference runs one phoneme batch and returns 24 kHz float samples.
func (k *KokoroTTS) runInference(phonemes, voice string, speed float32) ([]float32, error) {
	tokens := KokoroTokenize(phonemes)
	if len(tokens) == 0 {
		return nil, nil
	}
	if len(tokens) > kokoroMaxPhonemes {
		tokens = tokens[:kokoroMaxPhonemes]
	}
	style := k.voices[voice]
	if style == nil || len(style.Shape) < 1 {
		return nil, fmt.Errorf("voice %q not loaded", voice)
	}
	// style array is [510, 1, 256]; pick the row for this token count.
	row := len(tokens)
	if row >= style.Shape[0] {
		row = style.Shape[0] - 1
	}
	styleVec := style.Data[row*kokoroStyleDim : (row+1)*kokoroStyleDim]

	padded := make([]int64, 0, len(tokens)+2)
	padded = append(padded, 0)
	padded = append(padded, tokens...)
	padded = append(padded, 0)

	ids, err := k.session.IntTensorFor("input_ids", []int64{1, int64(len(padded))}, padded)
	if err != nil {
		return nil, err
	}
	st, err := ort.FloatTensor([]int64{1, kokoroStyleDim}, append([]float32(nil), styleVec...))
	if err != nil {
		ids.Destroy()
		return nil, err
	}
	sp, err := ort.FloatTensor([]int64{1}, []float32{speed})
	if err != nil {
		ids.Destroy()
		st.Destroy()
		return nil, err
	}
	inputs := map[string]onnx.Value{"input_ids": ids, "style": st, "speed": sp}
	outputs, err := k.session.RunNamed(inputs)
	ort.DestroyAll([]onnx.Value{ids, st, sp})
	if err != nil {
		return nil, err
	}
	defer ort.DestroyAll(outputs)
	samples, _, err := ort.FloatData(outputs[0])
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), samples...), nil
}

// SynthesizeIter yields PCM for each phoneme batch as its inference
// completes, so the device can start playing batch 1 while batch 2 is
// still inferring.
func (k *KokoroTTS) SynthesizeIter(text string, emit func(pcm []byte) error) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.session == nil {
		return fmt.Errorf("call Load() first")
	}
	raw, err := k.phonemizer.Phonemize(KokoroNormalizeText(text))
	if err != nil {
		return err
	}
	phonemes := kokoroPostProcess(raw, k.lang())
	for _, batch := range KokoroSplitPhonemes(phonemes) {
		samples, err := k.runInference(batch, k.voice(), 1.0)
		if err != nil {
			return err
		}
		if len(samples) == 0 {
			continue
		}
		pcm := dsp.Float32ToInt16(samples, 32767)
		if k.sampleRate != kokoroNativeRate {
			pcm = dsp.Resample(pcm, kokoroNativeRate, k.sampleRate)
		}
		if err := emit(dsp.Int16ToBytes(pcm)); err != nil {
			return err
		}
	}
	return nil
}

// Synthesize renders text to PCM at SampleRate.
func (k *KokoroTTS) Synthesize(text string) ([]byte, error) {
	return SynthesizeAll(k, text)
}
