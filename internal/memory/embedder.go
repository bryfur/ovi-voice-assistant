package memory

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"path/filepath"
	"strings"
	"sync"

	onnx "github.com/yalue/onnxruntime_go"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/download"
	"github.com/bryfur/ovi-voice-assistant/internal/ort"
)

// Embedder generates text embeddings.
type Embedder interface {
	// Embed embeds a batch of texts, returning one vector per input.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// EmbedOne embeds a single text.
	EmbedOne(ctx context.Context, text string) ([]float32, error)
}

// EmbedOneWith is a helper implementing EmbedOne via Embed.
func EmbedOneWith(ctx context.Context, e Embedder, text string) ([]float32, error) {
	vecs, err := e.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, nil
	}
	return vecs[0], nil
}

// ONNXEmbedder runs a sentence-transformers model (e.g. all-MiniLM-L6-v2)
// locally via ONNX Runtime.
type ONNXEmbedder struct {
	modelName string
	maxLen    int

	mu        sync.Mutex
	session   *ort.Session
	tokenizer *WordPieceTokenizer
}

// NewONNXEmbedder creates an unloaded embedder for a HuggingFace repo id.
func NewONNXEmbedder(model string) *ONNXEmbedder {
	if model == "" {
		model = "sentence-transformers/all-MiniLM-L6-v2"
	}
	return &ONNXEmbedder{modelName: model, maxLen: 256}
}

// modelRepos returns candidate HF repos providing onnx/model.onnx + vocab.txt.
func (e *ONNXEmbedder) modelRepos() []string {
	repos := []string{e.modelName}
	base := e.modelName[strings.LastIndex(e.modelName, "/")+1:]
	repos = append(repos, "Xenova/"+base, "sentence-transformers/"+base)
	return repos
}

// Load downloads the model on first use and creates the session.
func (e *ONNXEmbedder) Load() error {
	dir := filepath.Join(config.CacheDir(), "fastembed", strings.ReplaceAll(e.modelName, "/", "--"))
	var modelPath, vocabPath string
	var lastErr error
	for _, repo := range e.modelRepos() {
		mp, err := download.Ensure(filepath.Join(dir, "model.onnx"), download.HuggingFaceURL(repo, "onnx/model.onnx"))
		if err != nil {
			lastErr = err
			continue
		}
		vp, err := download.Ensure(filepath.Join(dir, "vocab.txt"), download.HuggingFaceURL(repo, "vocab.txt"))
		if err != nil {
			lastErr = err
			continue
		}
		modelPath, vocabPath = mp, vp
		break
	}
	if modelPath == "" {
		return fmt.Errorf("download embedding model %s: %w", e.modelName, lastErr)
	}
	tok, err := LoadWordPiece(vocabPath, e.maxLen)
	if err != nil {
		return err
	}
	sess, err := ort.NewSession(modelPath, ort.SessionConfig{IntraOpThreads: 2, InterOpThreads: 1})
	if err != nil {
		return err
	}
	e.mu.Lock()
	e.session = sess
	e.tokenizer = tok
	e.mu.Unlock()
	slog.Info("Embedding model loaded", "model", e.modelName)
	return nil
}

// Close frees the session.
func (e *ONNXEmbedder) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
}

// EmbedOne implements Embedder.
func (e *ONNXEmbedder) EmbedOne(ctx context.Context, text string) ([]float32, error) {
	return EmbedOneWith(ctx, e, text)
}

// Embed implements Embedder. Texts are embedded one at a time (batching
// with padding is unnecessary at voice-assistant volumes).
func (e *ONNXEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session == nil {
		return nil, fmt.Errorf("call Load() first")
	}
	out := make([][]float32, 0, len(texts))
	for _, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vec, err := e.embedOneLocked(text)
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

func (e *ONNXEmbedder) embedOneLocked(text string) ([]float32, error) {
	ids := e.tokenizer.Encode(text)
	n := int64(len(ids))
	mask := make([]int64, n)
	types := make([]int64, n)
	for i := range mask {
		mask[i] = 1
	}
	inputs := map[string]onnx.Value{}
	var toDestroy []onnx.Value
	defer func() { ort.DestroyAll(toDestroy) }()
	for _, name := range e.session.InputNames {
		var data []int64
		switch name {
		case "input_ids":
			data = ids
		case "attention_mask":
			data = mask
		case "token_type_ids":
			data = types
		default:
			return nil, fmt.Errorf("unexpected embedding model input %q", name)
		}
		t, err := e.session.IntTensorFor(name, []int64{1, n}, data)
		if err != nil {
			return nil, err
		}
		toDestroy = append(toDestroy, t)
		inputs[name] = t
	}
	outputs, err := e.session.RunNamed(inputs)
	if err != nil {
		return nil, err
	}
	defer ort.DestroyAll(outputs)
	data, shape, err := ort.FloatData(outputs[0])
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("unexpected embedding output shape %v", shape)
	}
	dim := int(shape[2])
	seq := int(shape[1])
	vec := make([]float32, dim)
	var count float32
	for t := 0; t < seq && t < len(mask); t++ {
		if mask[t] == 0 {
			continue
		}
		count++
		row := data[t*dim : (t+1)*dim]
		for i, v := range row {
			vec[i] += v
		}
	}
	if count > 0 {
		for i := range vec {
			vec[i] /= count
		}
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec, nil
}
