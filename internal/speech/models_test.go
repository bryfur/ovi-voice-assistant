package speech

import (
	"archive/tar"
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func tarball(files map[string]string) []byte {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for name, content := range files {
		tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(content)), Typeflag: tar.TypeReg})
		tw.Write([]byte(content))
	}
	tw.Close()
	return buf.Bytes()
}

func serve(t *testing.T, files map[string][]byte) *int {
	t.Helper()
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if data, ok := files[r.URL.Path]; ok {
			w.Write(data)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	oldURL, oldDir := modelBaseURL, modelDir
	modelBaseURL, modelDir = srv.URL+"/", t.TempDir()
	t.Cleanup(func() { modelBaseURL, modelDir = oldURL, oldDir })
	return &hits
}

func TestEnsureExtractsAndCaches(t *testing.T) {
	hits := serve(t, map[string][]byte{"/tts-models/pack.tar.bz2": tarball(map[string]string{
		"pack/model.int8.onnx": "m", "pack/espeak-ng-data/en_dict": "d", "pack/../evil": "x",
	})})

	dir, err := ensurePack(ttsRelease, "pack")
	dir2, err2 := ensurePack(ttsRelease, "pack")

	if err != nil || err2 != nil || dir != dir2 || *hits != 1 {
		t.Fatalf("dir=%s err=%v err2=%v hits=%d", dir, err, err2, *hits)
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "espeak-ng-data", "en_dict")); string(data) != "d" {
		t.Fatal("nested file not extracted")
	}
	if _, err := os.Stat(filepath.Join(modelDir, "evil")); err == nil {
		t.Fatal("path traversal entry must be skipped")
	}
	if m, err := findFile(dir, "model.int8.onnx", "model.onnx"); err != nil || filepath.Base(m) != "model.int8.onnx" {
		t.Fatalf("findFile = %q, %v", m, err)
	}
	if _, err := findFile(dir, "nope*.onnx"); err == nil {
		t.Fatal("expected findFile error")
	}
}

func TestEnsureMissingPack(t *testing.T) {
	serve(t, nil)

	if _, err := ensurePack(asrRelease, "missing"); err == nil {
		t.Fatal("expected error")
	}
	if _, err := os.Stat(filepath.Join(modelDir, ".tmp-missing")); err == nil {
		t.Fatal("temp dir must be cleaned up")
	}
}

func TestEnsureFile(t *testing.T) {
	hits := serve(t, map[string][]byte{"/asr-models/silero_vad.onnx": []byte("onnx")})

	p1, err := ensureFile(asrRelease, "silero_vad.onnx")
	p2, _ := ensureFile(asrRelease, "silero_vad.onnx")

	if err != nil || p1 != p2 || *hits != 1 {
		t.Fatalf("p1=%s err=%v hits=%d", p1, err, *hits)
	}
	if data, _ := os.ReadFile(p1); string(data) != "onnx" {
		t.Fatal("content wrong")
	}
}
