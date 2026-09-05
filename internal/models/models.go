// Package models fetches sherpa-onnx model packs into the local cache.
package models

import (
	"archive/tar"
	"bytes"
	"compress/bzip2"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Releases on github.com/k2-fsa/sherpa-onnx that host the packs.
const (
	ASR = "asr-models"
	TTS = "tts-models"
)

// BaseURL is the release download root; tests may override it.
var BaseURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"

// Dir is the local model cache.
var Dir = filepath.Join(config.CacheDir(), "models")

// Ensure returns the directory of an extracted pack, downloading
// <release>/<name>.tar.bz2 if it is not cached yet.
func Ensure(release, name string) (string, error) {
	dir := filepath.Join(Dir, name)
	if _, err := os.Stat(dir); err == nil {
		return dir, nil
	}
	slog.Info("Downloading model", "name", name)
	start := time.Now()
	body, err := get(BaseURL + release + "/" + name + ".tar.bz2")
	if err != nil {
		return "", err
	}
	defer body.Close()
	tmp := filepath.Join(Dir, ".tmp-"+name)
	os.RemoveAll(tmp)
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return "", err
	}
	if err := extract(body, tmp, name+"/"); err != nil {
		os.RemoveAll(tmp)
		return "", fmt.Errorf("extract %s: %w", name, err)
	}
	if err := os.Rename(tmp, dir); err != nil {
		return "", err
	}
	slog.Info("Model ready", "name", name, "secs", int(time.Since(start).Seconds()))
	return dir, nil
}

// EnsureFile returns the path of a single release asset, downloading it
// if needed.
func EnsureFile(release, name string) (string, error) {
	path := filepath.Join(Dir, name)
	if st, err := os.Stat(path); err == nil && st.Size() > 0 {
		return path, nil
	}
	if err := os.MkdirAll(Dir, 0o755); err != nil {
		return "", err
	}
	slog.Info("Downloading model", "name", name)
	body, err := get(BaseURL + release + "/" + name)
	if err != nil {
		return "", err
	}
	defer body.Close()
	f, err := os.Create(path + ".part")
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, body); err != nil {
		f.Close()
		os.Remove(path + ".part")
		return "", err
	}
	f.Close()
	return path, os.Rename(path+".part", path)
}

// Find returns the first file in dir matching one of the globs, in order.
func Find(dir string, globs ...string) (string, error) {
	for _, g := range globs {
		if m, _ := filepath.Glob(filepath.Join(dir, g)); len(m) > 0 {
			return m[0], nil
		}
	}
	return "", fmt.Errorf("no file matching %v in %s", globs, dir)
}

func get(url string) (io.ReadCloser, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("download %s: HTTP %s", url, resp.Status)
	}
	return resp.Body, nil
}

// extract unpacks a (bzip2-compressed or plain) tar stream into dir,
// stripping the leading prefix from entry names.
func extract(r io.Reader, dir, prefix string) error {
	br := newBufferedReader(r)
	var tr *tar.Reader
	if magic, _ := br.Peek(3); bytes.Equal(magic, []byte("BZh")) {
		tr = tar.NewReader(bzip2.NewReader(br))
	} else {
		tr = tar.NewReader(br)
	}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		name := strings.TrimPrefix(hdr.Name, "./")
		name = strings.TrimPrefix(name, prefix)
		if name == "" || strings.Contains(name, "..") {
			continue
		}
		path := filepath.Join(dir, name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(path, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		}
	}
}
