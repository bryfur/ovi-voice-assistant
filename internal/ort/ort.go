// Package ort wraps ONNX Runtime initialisation and session helpers.
//
// The onnxruntime shared library is not bundled with Go; it is located via
// OVI_ONNXRUNTIME_LIB, a system install, or downloaded from the official
// GitHub release into ~/.cache/ovi/onnxruntime on first use.
package ort

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	onnx "github.com/yalue/onnxruntime_go"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Version is the ONNX Runtime release downloaded when none is installed.
// It must match the C API headers bundled with the onnxruntime_go binding
// (ORT_API_VERSION 29 ↔ onnxruntime 1.29.x); other releases may fail to
// load or misbehave.
const Version = "1.29.0"

var (
	initOnce sync.Once
	initErr  error
)

// LibraryPath returns the resolved shared library path, downloading the
// runtime if necessary.
//
// Resolution order: OVI_ONNXRUNTIME_LIB, the cached download for Version,
// then the official GitHub release archive. System-wide installs are not
// probed because they rarely match the exact version the binding expects.
func LibraryPath() (string, error) {
	if p := os.Getenv("OVI_ONNXRUNTIME_LIB"); p != "" {
		return p, nil
	}
	libName := "libonnxruntime.so"
	switch runtime.GOOS {
	case "darwin":
		libName = "libonnxruntime.dylib"
	case "windows":
		libName = "onnxruntime.dll"
	}
	dir := filepath.Join(config.CacheDir(), "onnxruntime", Version)
	local := filepath.Join(dir, libName)
	if _, err := os.Stat(local); err == nil {
		return local, nil
	}
	return downloadRuntime(dir, libName)
}

func releaseAsset() (string, error) {
	arch := runtime.GOARCH
	switch runtime.GOOS {
	case "linux":
		switch arch {
		case "amd64":
			return "onnxruntime-linux-x64-" + Version + ".tgz", nil
		case "arm64":
			return "onnxruntime-linux-aarch64-" + Version + ".tgz", nil
		}
	case "darwin":
		switch arch {
		case "amd64":
			return "onnxruntime-osx-x86_64-" + Version + ".tgz", nil
		case "arm64":
			return "onnxruntime-osx-arm64-" + Version + ".tgz", nil
		}
	}
	return "", fmt.Errorf("no prebuilt onnxruntime for %s/%s; set OVI_ONNXRUNTIME_LIB", runtime.GOOS, arch)
}

func downloadRuntime(dir, libName string) (string, error) {
	asset, err := releaseAsset()
	if err != nil {
		return "", err
	}
	url := os.Getenv("OVI_ONNXRUNTIME_URL")
	if url == "" {
		url = "https://github.com/microsoft/onnxruntime/releases/download/v" + Version + "/" + asset
	}
	slog.Info("Downloading ONNX Runtime", "url", url)
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("download onnxruntime: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download onnxruntime: HTTP %s", resp.Status)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return "", err
	}
	tr := tar.NewReader(gz)
	var found string
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		base := filepath.Base(hdr.Name)
		if !strings.Contains(hdr.Name, "/lib/") || !strings.HasPrefix(base, strings.TrimSuffix(libName, filepath.Ext(libName))) {
			continue
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		dest := filepath.Join(dir, base)
		f, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return "", err
		}
		if _, err := io.Copy(f, tr); err != nil {
			f.Close()
			return "", err
		}
		f.Close()
		if base == libName {
			found = dest
		} else if found == "" && strings.HasPrefix(base, libName) {
			found = dest
		}
	}
	if found == "" {
		return "", fmt.Errorf("onnxruntime archive did not contain %s", libName)
	}
	// The unversioned name may be a symlink in the archive; make sure it exists.
	local := filepath.Join(dir, libName)
	if _, err := os.Stat(local); err != nil {
		data, err := os.ReadFile(found)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(local, data, 0o755); err != nil {
			return "", err
		}
	}
	slog.Info("ONNX Runtime installed", "path", local)
	return local, nil
}

// Init initialises the ONNX Runtime environment once.
func Init() error {
	initOnce.Do(func() {
		if onnx.IsInitialized() {
			return
		}
		path, err := LibraryPath()
		if err != nil {
			initErr = err
			return
		}
		onnx.SetSharedLibraryPath(path)
		if err := onnx.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("initialise onnxruntime (%s): %w", path, err)
			return
		}
		slog.Debug("ONNX Runtime initialised", "lib", path)
	})
	return initErr
}

// Provider selection.
type Provider string

const (
	ProviderCPU    Provider = "CPUExecutionProvider"
	ProviderCUDA   Provider = "CUDAExecutionProvider"
	ProviderCoreML Provider = "CoreMLExecutionProvider"
	ProviderDML    Provider = "DmlExecutionProvider"
)

// SessionConfig tunes a session.
type SessionConfig struct {
	IntraOpThreads int
	InterOpThreads int
	Parallel       bool
	OptExtended    bool // ORT_ENABLE_EXTENDED instead of ORT_ENABLE_ALL
	Providers      []Provider
	CoreMLCacheDir string
}

// Session is a dynamic-shape ONNX session with named inputs and outputs.
type Session struct {
	sess        *onnx.DynamicAdvancedSession
	Inputs      []onnx.InputOutputInfo
	Outputs     []onnx.InputOutputInfo
	InputNames  []string
	OutputNames []string
	Providers   []Provider
}

// NewSession loads a model with the given config. Input/output metadata is
// read from the model file so callers can build tensors of the right type.
func NewSession(path string, cfg SessionConfig) (*Session, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	opts, err := onnx.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	_ = opts.SetLogSeverityLevel(onnx.LoggingLevelError)
	// Always size the thread pool explicitly: ONNX Runtime only pins thread
	// affinity when the count is left at its default, which fails noisily on
	// hosts with a restricted CPU mask.
	if cfg.IntraOpThreads <= 0 {
		cfg.IntraOpThreads = runtime.NumCPU()
	}
	if cfg.InterOpThreads <= 0 {
		cfg.InterOpThreads = 1
	}
	if cfg.OptExtended {
		_ = opts.SetGraphOptimizationLevel(onnx.GraphOptimizationLevelEnableExtended)
	} else {
		_ = opts.SetGraphOptimizationLevel(onnx.GraphOptimizationLevelEnableAll)
	}
	if cfg.Parallel {
		_ = opts.SetExecutionMode(onnx.ExecutionModeParallel)
	} else {
		_ = opts.SetExecutionMode(onnx.ExecutionModeSequential)
	}
	_ = opts.SetIntraOpNumThreads(cfg.IntraOpThreads)
	_ = opts.SetInterOpNumThreads(cfg.InterOpThreads)
	// Inspect the model with the same options so the probe session does not
	// spin up a default-sized, affinity-pinned thread pool.
	inputs, outputs, err := onnx.GetInputOutputInfoWithOptions(path, opts)
	if err != nil {
		return nil, fmt.Errorf("inspect %s: %w", filepath.Base(path), err)
	}
	var used []Provider
	for _, p := range cfg.Providers {
		switch p {
		case ProviderCUDA:
			co, err := onnx.NewCUDAProviderOptions()
			if err != nil {
				continue
			}
			if err := opts.AppendExecutionProviderCUDA(co); err == nil {
				used = append(used, p)
			}
			co.Destroy()
		case ProviderCoreML:
			m := map[string]string{
				"ModelFormat":              "MLProgram",
				"MLComputeUnits":           "ALL",
				"RequireStaticInputShapes": "0",
			}
			if cfg.CoreMLCacheDir != "" {
				m["ModelCacheDirectory"] = cfg.CoreMLCacheDir
			}
			if err := opts.AppendExecutionProviderCoreMLV2(m); err == nil {
				used = append(used, p)
			}
		case ProviderDML:
			if err := opts.AppendExecutionProviderDirectML(0); err == nil {
				used = append(used, p)
			}
		}
	}
	used = append(used, ProviderCPU)

	inNames := make([]string, len(inputs))
	for i, in := range inputs {
		inNames[i] = in.Name
	}
	outNames := make([]string, len(outputs))
	for i, out := range outputs {
		outNames[i] = out.Name
	}
	sess, err := onnx.NewDynamicAdvancedSession(path, inNames, outNames, opts)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", filepath.Base(path), err)
	}
	return &Session{
		sess:        sess,
		Inputs:      inputs,
		Outputs:     outputs,
		InputNames:  inNames,
		OutputNames: outNames,
		Providers:   used,
	}, nil
}

// Run executes the model. Inputs must be in model order. Outputs are
// allocated by the runtime and must be destroyed by the caller.
func (s *Session) Run(inputs []onnx.Value) ([]onnx.Value, error) {
	outputs := make([]onnx.Value, len(s.OutputNames))
	if err := s.sess.Run(inputs, outputs); err != nil {
		return nil, err
	}
	return outputs, nil
}

// RunNamed executes the model with inputs looked up by name.
func (s *Session) RunNamed(inputs map[string]onnx.Value) ([]onnx.Value, error) {
	ordered := make([]onnx.Value, len(s.InputNames))
	for i, name := range s.InputNames {
		v, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("missing input %q", name)
		}
		ordered[i] = v
	}
	return s.Run(ordered)
}

// InputType returns the element type of a named input.
func (s *Session) InputType(name string) onnx.TensorElementDataType {
	for _, in := range s.Inputs {
		if in.Name == name {
			return in.DataType
		}
	}
	return onnx.TensorElementDataTypeUndefined
}

// HasInput reports whether the model declares the input.
func (s *Session) HasInput(name string) bool {
	for _, in := range s.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}

// Destroy frees the session.
func (s *Session) Destroy() {
	if s.sess != nil {
		_ = s.sess.Destroy()
		s.sess = nil
	}
}

// DestroyAll destroys a slice of values, ignoring nils.
func DestroyAll(values []onnx.Value) {
	for _, v := range values {
		if v != nil {
			_ = v.Destroy()
		}
	}
}

// FloatTensor creates a float32 tensor.
func FloatTensor(shape []int64, data []float32) (*onnx.Tensor[float32], error) {
	return onnx.NewTensor(onnx.Shape(shape), data)
}

// Int64Tensor creates an int64 tensor.
func Int64Tensor(shape []int64, data []int64) (*onnx.Tensor[int64], error) {
	return onnx.NewTensor(onnx.Shape(shape), data)
}

// Int32Tensor creates an int32 tensor.
func Int32Tensor(shape []int64, data []int32) (*onnx.Tensor[int32], error) {
	return onnx.NewTensor(onnx.Shape(shape), data)
}

// IntTensorFor creates an integer tensor of the element type the model
// expects for the named input (int32 or int64).
func (s *Session) IntTensorFor(name string, shape []int64, data []int64) (onnx.Value, error) {
	if s.InputType(name) == onnx.TensorElementDataTypeInt32 {
		d := make([]int32, len(data))
		for i, v := range data {
			d[i] = int32(v)
		}
		return Int32Tensor(shape, d)
	}
	return Int64Tensor(shape, data)
}

// FloatData extracts float32 data from an output value.
func FloatData(v onnx.Value) ([]float32, []int64, error) {
	t, ok := v.(*onnx.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("output is %T, not float32 tensor", v)
	}
	return t.GetData(), []int64(t.GetShape()), nil
}

// IntData extracts integer data (int32 or int64) from an output value.
func IntData(v onnx.Value) ([]int64, error) {
	switch t := v.(type) {
	case *onnx.Tensor[int64]:
		return t.GetData(), nil
	case *onnx.Tensor[int32]:
		d := t.GetData()
		out := make([]int64, len(d))
		for i, x := range d {
			out[i] = int64(x)
		}
		return out, nil
	}
	return nil, fmt.Errorf("output is %T, not an integer tensor", v)
}

// SelectProviders picks execution providers for a device preference:
// "cuda" forces CUDA, "auto" tries CUDA first, anything else uses the
// platform accelerator (CoreML on macOS, DirectML on Windows) if allowed.
// The CPU provider is always appended as a fallback.
func SelectProviders(device string, allowCoreML bool) []Provider {
	if strings.EqualFold(device, "cuda") {
		return []Provider{ProviderCUDA}
	}
	var ps []Provider
	if strings.EqualFold(device, "auto") {
		ps = append(ps, ProviderCUDA)
	}
	if runtime.GOOS == "darwin" && allowCoreML {
		ps = append(ps, ProviderCoreML)
	}
	if runtime.GOOS == "windows" {
		ps = append(ps, ProviderDML)
	}
	return ps
}
