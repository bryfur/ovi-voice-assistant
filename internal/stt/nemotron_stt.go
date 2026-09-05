package stt

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	onnx "github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/dsp/fourier"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/download"
	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
	"github.com/bryfur/ovi-voice-assistant/internal/ort"
)

// ── Mel spectrogram ─────────────────────────────────────────────

const (
	nemSampleRate = 16000
	nemNFFT       = 512
	nemHop        = 160
	nemWin        = 400
	nemMels       = 128
	nemPreemph    = 0.97
	nemLogGuard   = 5.960464477539063e-08 // 2**-24
)

// ── Encoder (cache-aware streaming FastConformer) ────────────────

const (
	melShift       = 56 // new mel frames per encoder chunk (560 ms)
	preEncodeCache = 9  // mel context frames prepended to each chunk
	encLayers      = 24
	encDim         = 1024
	cacheChDim     = 70
	cacheTimeDim   = 8
)

// ── Decoder (RNNT prediction network + joint) ───────────────────

const (
	blankID            = 1024
	maxSymbolsPerFrame = 10
	predHidden         = 640
)

// ── Model source ────────────────────────────────────────────────

const (
	nemotronRepo    = "danielbodart/nemotron-speech-600m-onnx"
	nemotronDefault = "int8-dynamic"
)

var nemotronVariants = map[string]bool{"fp32": true, "fp16": true, "int8-dynamic": true, "int8-static": true}

// NemotronSTT is NVIDIA Nemotron Speech 600M with direct ONNX Runtime
// inference. Mel spectrogram extraction, the cache-aware streaming encoder
// and RNNT greedy decoding are all implemented here.
type NemotronSTT struct {
	settings *config.Settings
	params   ListenParams

	encoder *ort.Session
	decoder *ort.Session
	fb      []float64 // [nemMels * (nemNFFT/2+1)] filterbank, float64
	hann    []float64 // nemNFFT window (zero-padded symmetric Hann)
	tokens  []string
	vad     Prober

	fftPool sync.Pool

	// LoadVAD constructs the VAD; tests may override it.
	LoadVAD func() (Prober, error)
}

// NewNemotronSTT creates an unloaded provider.
func NewNemotronSTT(settings *config.Settings) *NemotronSTT {
	p := DefaultListenParams(settings.Mic.SampleRate, settings.Mic.SampleWidth)
	p.SilenceTimeout = 750 * time.Millisecond
	return &NemotronSTT{
		settings: settings,
		params:   p,
		LoadVAD:  func() (Prober, error) { return LoadSileroVAD() },
	}
}

// Load downloads the model files and creates the ONNX sessions.
func (n *NemotronSTT) Load() error {
	variant := n.settings.STT.Model
	if !nemotronVariants[variant] {
		variant = nemotronDefault
	}
	slog.Info("Loading Nemotron Speech", "variant", variant)

	dir := filepath.Join(config.CacheDir(), "nemotron")
	files := []string{
		variant + "/encoder_model.onnx", variant + "/encoder_model.onnx.data",
		variant + "/decoder_model.onnx", variant + "/decoder_model.onnx.data",
		"shared/filterbank.bin", "shared/tokens.txt",
	}
	paths := map[string]string{}
	for _, f := range files {
		p, err := download.Ensure(filepath.Join(dir, f), download.HuggingFaceURL(nemotronRepo, f))
		if err != nil {
			return err
		}
		paths[f] = p
	}

	providers := ort.SelectProviders(n.settings.STT.Device, false)
	total := runtime.NumCPU()
	encThreads := max(1, min(8, total/2))
	enc, err := ort.NewSession(paths[variant+"/encoder_model.onnx"], ort.SessionConfig{
		IntraOpThreads: encThreads, InterOpThreads: 1, Providers: providers,
	})
	if err != nil {
		return err
	}
	dec, err := ort.NewSession(paths[variant+"/decoder_model.onnx"], ort.SessionConfig{
		IntraOpThreads: 2, InterOpThreads: 1, Providers: providers,
	})
	if err != nil {
		enc.Destroy()
		return err
	}
	slog.Info("Nemotron ORT", "providers", enc.Providers, "encoder_threads", encThreads, "decoder_threads", 2)

	fbBytes, err := os.ReadFile(paths["shared/filterbank.bin"])
	if err != nil {
		return err
	}
	if len(fbBytes) != nemMels*(nemNFFT/2+1)*4 {
		return fmt.Errorf("filterbank.bin has unexpected size %d", len(fbBytes))
	}
	n.fb = make([]float64, nemMels*(nemNFFT/2+1))
	for i := range n.fb {
		n.fb[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(fbBytes[i*4:])))
	}

	tokFile, err := os.Open(paths["shared/tokens.txt"])
	if err != nil {
		return err
	}
	defer tokFile.Close()
	n.tokens = nil
	scanner := bufio.NewScanner(tokFile)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if i := strings.LastIndex(line, " "); i >= 0 {
			line = line[:i]
		}
		n.tokens = append(n.tokens, line)
	}

	// Symmetric Hann window zero-padded to N_FFT, kept as float64 so the
	// entire mel pipeline runs in float64 (matches reference).
	n.hann = make([]float64, nemNFFT)
	wo := (nemNFFT - nemWin) / 2
	copy(n.hann[wo:], dsp.HannWindow(nemWin))

	n.encoder, n.decoder = enc, dec
	n.fftPool = sync.Pool{New: func() any { return fourier.NewFFT(nemNFFT) }}

	vad, err := n.LoadVAD()
	if err != nil {
		return err
	}
	n.vad = vad
	slog.Info("Nemotron STT ready (Silero VAD)")
	return nil
}

// Close frees the sessions.
func (n *NemotronSTT) Close() {
	if n.encoder != nil {
		n.encoder.Destroy()
	}
	if n.decoder != nil {
		n.decoder.Destroy()
	}
}

// ── mel spectrogram ──────────────────────────────────────────

// preEmphasis applies y[t] = x[t] - 0.97*x[t-1] with a carried previous sample.
func preEmphasis(samples []float64, prev float64) []float64 {
	out := make([]float64, len(samples))
	for i, s := range samples {
		out[i] = s - nemPreemph*prev
		prev = s
	}
	return out
}

// melFrames computes n mel frames [nemMels, n] starting at frame index start
// from pre-emphasised audio, using reflect indexing at the left edge and
// clipping at the right. Output is column-major: frame f at out[f*nemMels:].
func (n *NemotronSTT) melFrames(audio []float64, start, count int) []float32 {
	if count <= 0 {
		return nil
	}
	fft := n.fftPool.Get().(*fourier.FFT)
	defer n.fftPool.Put(fft)
	pad := nemNFFT / 2
	nbins := nemNFFT/2 + 1
	frame := make([]float64, nemNFFT)
	coeffs := make([]complex128, nbins)
	power := make([]float64, nbins)
	out := make([]float32, count*nemMels)
	for f := 0; f < count; f++ {
		center := (start + f) * nemHop
		for i := 0; i < nemNFFT; i++ {
			idx := center - pad + i
			if idx < 0 {
				idx = -idx
			}
			if idx >= len(audio) {
				idx = 2*(len(audio)-1) - idx // numpy reflect padding
				if idx < 0 {
					idx = 0
				}
			}
			frame[i] = audio[idx] * n.hann[i]
		}
		power = dsp.PowerSpectrum(fft, frame, coeffs, power)
		for m := 0; m < nemMels; m++ {
			row := n.fb[m*nbins : (m+1)*nbins]
			var acc float64
			for k, p := range power {
				acc += p * row[k]
			}
			out[f*nemMels+m] = float32(math.Log(acc + nemLogGuard))
		}
	}
	return out
}

// ── encoder / decoder state ──────────────────────────────────

type encState struct {
	cacheCh    []float32 // [1,24,70,1024]
	cacheTime  []float32 // [1,24,1024,8]
	cacheChLen int64
	preCache   []float32 // [nemMels * preEncodeCache] column-major frames
	s1, s2     []float32 // [2,1,640]
	lastTok    int64
	tokens     []int64
}

func newEncState() *encState {
	return &encState{
		cacheCh:   make([]float32, encLayers*cacheChDim*encDim),
		cacheTime: make([]float32, encLayers*encDim*cacheTimeDim),
		preCache:  make([]float32, nemMels*preEncodeCache),
		s1:        make([]float32, 2*predHidden),
		s2:        make([]float32, 2*predHidden),
	}
}

// toRowMajor converts column-major frames [frames][mels] into [mels][frames].
func toRowMajor(frames []float32, count int) []float32 {
	out := make([]float32, nemMels*count)
	for f := 0; f < count; f++ {
		for m := 0; m < nemMels; m++ {
			out[m*count+f] = frames[f*nemMels+m]
		}
	}
	return out
}

// decodeChunks processes complete melShift chunks of mel (column-major
// frames) starting at *cursor, appending tokens to st.
func (n *NemotronSTT) decodeChunks(mel []float32, nFrames int, cursor *int, st *encState) error {
	for *cursor+melShift <= nFrames {
		chunk := mel[*cursor*nemMels : (*cursor+melShift)*nemMels]
		// encoder input: [preEncodeCache + melShift] frames
		frames := make([]float32, 0, (preEncodeCache+melShift)*nemMels)
		frames = append(frames, st.preCache...)
		frames = append(frames, chunk...)
		total := preEncodeCache + melShift
		signal := toRowMajor(frames, total)

		if err := n.runEncoderChunk(signal, total, st); err != nil {
			return err
		}
		copy(st.preCache, chunk[(melShift-preEncodeCache)*nemMels:])
		*cursor += melShift
	}
	return nil
}

func (n *NemotronSTT) runEncoderChunk(signal []float32, nFrames int, st *encState) error {
	en := n.encoder.InputNames
	if len(en) < 5 {
		return fmt.Errorf("encoder has %d inputs, expected 5", len(en))
	}
	var toDestroy []onnx.Value
	defer func() { ort.DestroyAll(toDestroy) }()

	sig, err := ort.FloatTensor([]int64{1, nemMels, int64(nFrames)}, signal)
	if err != nil {
		return err
	}
	length, err := n.encoder.IntTensorFor(en[1], []int64{1}, []int64{int64(nFrames)})
	if err != nil {
		return err
	}
	cch, err := ort.FloatTensor([]int64{1, encLayers, cacheChDim, encDim}, st.cacheCh)
	if err != nil {
		return err
	}
	ctm, err := ort.FloatTensor([]int64{1, encLayers, encDim, cacheTimeDim}, st.cacheTime)
	if err != nil {
		return err
	}
	clen, err := n.encoder.IntTensorFor(en[4], []int64{1}, []int64{st.cacheChLen})
	if err != nil {
		return err
	}
	toDestroy = append(toDestroy, sig, length, cch, ctm, clen)

	outputs, err := n.encoder.Run([]onnx.Value{sig, length, cch, ctm, clen})
	if err != nil {
		return err
	}
	defer ort.DestroyAll(outputs)
	if len(outputs) < 5 {
		return fmt.Errorf("encoder returned %d outputs", len(outputs))
	}
	encOut, shape, err := ort.FloatData(outputs[0])
	if err != nil {
		return err
	}
	encLenArr, err := ort.IntData(outputs[1])
	if err != nil {
		return err
	}
	encLen := int(encLenArr[0])
	newCh, _, _ := ort.FloatData(outputs[2])
	newTime, _, _ := ort.FloatData(outputs[3])
	newLen, _ := ort.IntData(outputs[4])
	copy(st.cacheCh, newCh)
	copy(st.cacheTime, newTime)
	if len(newLen) > 0 {
		st.cacheChLen = newLen[0]
	}
	// encOut is [1, encDim, T]
	if len(shape) != 3 {
		return fmt.Errorf("unexpected encoder output shape %v", shape)
	}
	T := int(shape[2])
	frame := make([]float32, encDim)
	for t := 0; t < encLen && t < T; t++ {
		for d := 0; d < encDim; d++ {
			frame[d] = encOut[d*T+t]
		}
		if err := n.decodeFrame(frame, st); err != nil {
			return err
		}
	}
	return nil
}

// decodeFrame runs greedy RNNT decoding for one encoder frame.
func (n *NemotronSTT) decodeFrame(frame []float32, st *encState) error {
	dn := n.decoder.InputNames
	if len(dn) < 5 {
		return fmt.Errorf("decoder has %d inputs, expected 5", len(dn))
	}
	for i := 0; i < maxSymbolsPerFrame; i++ {
		ef, err := ort.FloatTensor([]int64{1, encDim, 1}, frame)
		if err != nil {
			return err
		}
		tgt, err := n.decoder.IntTensorFor(dn[1], []int64{1, 1}, []int64{st.lastTok})
		if err != nil {
			ef.Destroy()
			return err
		}
		tgtLen, err := n.decoder.IntTensorFor(dn[2], []int64{1}, []int64{1})
		if err != nil {
			ef.Destroy()
			tgt.Destroy()
			return err
		}
		s1, err := ort.FloatTensor([]int64{2, 1, predHidden}, st.s1)
		if err != nil {
			return err
		}
		s2, err := ort.FloatTensor([]int64{2, 1, predHidden}, st.s2)
		if err != nil {
			return err
		}
		inputs := []onnx.Value{ef, tgt, tgtLen, s1, s2}
		outputs, err := n.decoder.Run(inputs)
		ort.DestroyAll(inputs)
		if err != nil {
			return err
		}
		logits, _, err := ort.FloatData(outputs[0])
		if err != nil {
			ort.DestroyAll(outputs)
			return err
		}
		tid := argmax(logits)
		if tid == blankID || len(outputs) < 4 {
			ort.DestroyAll(outputs)
			break // do NOT update LSTM states on blank
		}
		ns1, _, _ := ort.FloatData(outputs[2])
		ns2, _, _ := ort.FloatData(outputs[3])
		copy(st.s1, ns1)
		copy(st.s2, ns2)
		ort.DestroyAll(outputs)
		st.tokens = append(st.tokens, int64(tid))
		st.lastTok = int64(tid)
	}
	return nil
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func (n *NemotronSTT) tokensToText(ids []int64) string {
	if len(ids) == 0 || n.tokens == nil {
		return ""
	}
	var sb strings.Builder
	for _, t := range ids {
		if t >= 0 && int(t) < len(n.tokens) {
			sb.WriteString(n.tokens[t])
		}
	}
	return strings.TrimSpace(strings.ReplaceAll(sb.String(), "▁", " "))
}

// ── public: batch transcribe ─────────────────────────────────

// Transcribe transcribes a complete PCM buffer.
func (n *NemotronSTT) Transcribe(pcm []byte) (string, error) {
	if n.encoder == nil {
		return "", fmt.Errorf("call Load() first")
	}
	samples := dsp.BytesToInt16(pcm)
	if len(samples) < n.settings.Mic.SampleRate/10 {
		return "", nil
	}
	audio := make([]float64, len(samples))
	for i, s := range samples {
		audio[i] = float64(s) / 32768.0
	}
	pre := preEmphasis(audio, 0)
	// np.pad(..., reflect) on both sides then framing == centred frames with
	// reflect indexing, which melFrames provides.
	nFrames := len(pre)/nemHop + 1
	mel := n.melFrames(pre, 0, nFrames)
	st := newEncState()
	cursor := 0
	if err := n.decodeChunks(mel, nFrames, &cursor, st); err != nil {
		return "", err
	}
	text := n.tokensToText(st.tokens)
	if text != "" {
		slog.Debug("Transcribed", "text", text)
	}
	return text, nil
}

// ── public: streaming transcribe ─────────────────────────────

// TranscribeStream accumulates audio with Silero VAD, runs encoder chunks
// as mel frames become available, and decodes tokens incrementally.
func (n *NemotronSTT) TranscribeStream(ctx context.Context, chunks <-chan []byte, onVADStart VADStartCallback) (string, error) {
	if n.encoder == nil || n.vad == nil {
		return "", fmt.Errorf("call Load() first")
	}
	listener := NewVADListener(n.vad, n.params, onVADStart)
	pad := nemNFFT / 2

	var (
		preemphPrev float64
		audioBuf    []float64 // pre-emphasised audio
		mel         []float32 // column-major mel frames
		melComputed int
		cursor      int
		st          = newEncState()
		lastPartial string
	)
	reset := func() {
		preemphPrev = 0
		audioBuf = audioBuf[:0]
		mel = mel[:0]
		melComputed = 0
		cursor = 0
		st = newEncState()
	}
	appendAudio := func(chunk []byte) {
		samples := dsp.BytesToInt16(chunk)
		if len(samples) == 0 {
			return
		}
		f := make([]float64, len(samples))
		for i, s := range samples {
			f[i] = float64(s) / 32768.0
		}
		audioBuf = append(audioBuf, preEmphasis(f, preemphPrev)...)
		preemphPrev = f[len(f)-1]
	}
	advance := func() error {
		if !listener.SpeechDetected() || len(audioBuf) <= pad {
			return nil
		}
		avail := (len(audioBuf)-pad)/nemHop + 1
		if newFrames := avail - melComputed; newFrames > 0 {
			mel = append(mel, n.melFrames(audioBuf, melComputed, newFrames)...)
			melComputed = avail
		}
		if err := n.decodeChunks(mel, melComputed, &cursor, st); err != nil {
			return err
		}
		if partial := n.tokensToText(st.tokens); partial != "" && partial != lastPartial {
			slog.Debug("Partial", "text", partial)
			lastPartial = partial
		}
		return nil
	}

loop:
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case chunk, ok := <-chunks:
			if !ok {
				break loop
			}
			res, err := listener.Feed(chunk)
			if err != nil {
				return "", err
			}
			switch res {
			case FeedGiveUp:
				break loop
			case FeedEndOfSpeech:
				appendAudio(chunk)
				break loop
			case FeedResetSpeech:
				reset()
				continue
			}
			appendAudio(chunk)
			if err := advance(); err != nil {
				return "", err
			}
		}
	}

	// Final flush: process remaining mel frames
	if listener.SpeechDetected() && len(audioBuf) > pad {
		avail := (len(audioBuf)-pad)/nemHop + 1
		if newFrames := avail - melComputed; newFrames > 0 {
			mel = append(mel, n.melFrames(audioBuf, melComputed, newFrames)...)
			melComputed = avail
		}
		if remaining := melComputed - cursor; remaining > 0 && remaining < melShift {
			mel = append(mel, make([]float32, (melShift-remaining)*nemMels)...)
			melComputed += melShift - remaining
		}
		if err := n.decodeChunks(mel, melComputed, &cursor, st); err != nil {
			return "", err
		}
	}
	text := n.tokensToText(st.tokens)
	if text != "" {
		slog.Debug("Transcribed", "text", text)
	}
	return text, nil
}
