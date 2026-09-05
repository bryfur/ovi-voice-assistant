package stt

import (
	"fmt"
	"path/filepath"
	"sync"

	onnx "github.com/yalue/onnxruntime_go"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/download"
	"github.com/bryfur/ovi-voice-assistant/internal/ort"
)

// Silero VAD constants.
const (
	VADChunkSamples = 512 // Silero expects 512-sample chunks at 16kHz
	vadContext      = 64  // context samples carried between chunks
	vadStateDim     = 128
	vadSampleRate   = 16000
)

// SileroModelURL is the official Silero VAD v5 ONNX model.
var SileroModelURL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

// SileroVAD wraps the Silero VAD ONNX model.
type SileroVAD struct {
	mu      sync.Mutex
	session *ort.Session
	useHC   bool // model exposes h/c inputs instead of a single state
}

// LoadSileroVAD downloads (if needed) and loads the VAD model.
func LoadSileroVAD() (*SileroVAD, error) {
	path, err := download.Ensure(filepath.Join(config.CacheDir(), "silero", "silero_vad.onnx"), SileroModelURL)
	if err != nil {
		return nil, err
	}
	sess, err := ort.NewSession(path, ort.SessionConfig{IntraOpThreads: 1, InterOpThreads: 1})
	if err != nil {
		return nil, err
	}
	v := &SileroVAD{session: sess}
	if sess.HasInput("h") && sess.HasInput("c") {
		v.useHC = true
	} else if !sess.HasInput("state") {
		sess.Destroy()
		return nil, fmt.Errorf("unrecognised Silero VAD model inputs: %v", sess.InputNames)
	}
	return v, nil
}

// Close frees the model.
func (v *SileroVAD) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.session != nil {
		v.session.Destroy()
		v.session = nil
	}
}

// VADState is the recurrent state for one audio stream.
type VADState struct {
	state   []float32 // [2,1,128] (or h ++ c for the h/c variant)
	context []float32 // last 64 samples
}

// NewState creates a zeroed state.
func (v *SileroVAD) NewState() *VADState {
	return &VADState{
		state:   make([]float32, 2*vadStateDim),
		context: make([]float32, vadContext),
	}
}

// Probability runs one 512-sample chunk (float32 in [-1,1]) and returns the
// speech probability, updating st in place.
func (v *SileroVAD) Probability(st *VADState, samples []float32) (float32, error) {
	if len(samples) != VADChunkSamples {
		return 0, fmt.Errorf("vad: expected %d samples, got %d", VADChunkSamples, len(samples))
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.session == nil {
		return 0, fmt.Errorf("vad: closed")
	}
	input := make([]float32, 0, vadContext+VADChunkSamples)
	input = append(input, st.context...)
	input = append(input, samples...)
	copy(st.context, samples[len(samples)-vadContext:])

	inputs := map[string]onnx.Value{}
	var toDestroy []onnx.Value
	defer func() { ort.DestroyAll(toDestroy) }()

	inT, err := ort.FloatTensor([]int64{1, int64(len(input))}, input)
	if err != nil {
		return 0, err
	}
	toDestroy = append(toDestroy, inT)
	inputs["input"] = inT

	if v.useHC {
		h, err := ort.FloatTensor([]int64{1, 1, vadStateDim}, append([]float32(nil), st.state[:vadStateDim]...))
		if err != nil {
			return 0, err
		}
		c, err := ort.FloatTensor([]int64{1, 1, vadStateDim}, append([]float32(nil), st.state[vadStateDim:]...))
		if err != nil {
			return 0, err
		}
		toDestroy = append(toDestroy, h, c)
		inputs["h"] = h
		inputs["c"] = c
	} else {
		stT, err := ort.FloatTensor([]int64{2, 1, vadStateDim}, append([]float32(nil), st.state...))
		if err != nil {
			return 0, err
		}
		toDestroy = append(toDestroy, stT)
		inputs["state"] = stT
	}
	if v.session.HasInput("sr") {
		sr, err := onnx.NewScalar[int64](vadSampleRate)
		if err != nil {
			return 0, err
		}
		toDestroy = append(toDestroy, sr)
		inputs["sr"] = sr
	}

	outputs, err := v.session.RunNamed(inputs)
	if err != nil {
		return 0, err
	}
	defer ort.DestroyAll(outputs)

	prob, _, err := ort.FloatData(outputs[0])
	if err != nil {
		return 0, err
	}
	if v.useHC {
		if len(outputs) >= 3 {
			h, _, _ := ort.FloatData(outputs[1])
			c, _, _ := ort.FloatData(outputs[2])
			copy(st.state[:vadStateDim], h)
			copy(st.state[vadStateDim:], c)
		}
	} else if len(outputs) >= 2 {
		s, _, _ := ort.FloatData(outputs[1])
		copy(st.state, s)
	}
	if len(prob) == 0 {
		return 0, fmt.Errorf("vad: empty output")
	}
	return prob[0], nil
}
