package download

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureDownloadsOnce(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Write([]byte("model-bytes"))
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "sub", "model.onnx")

	p1, err1 := Ensure(dest, srv.URL)
	p2, err2 := Ensure(dest, srv.URL)

	if err1 != nil || err2 != nil || p1 != dest || p2 != dest {
		t.Fatalf("got %q %v / %q %v", p1, err1, p2, err2)
	}
	if hits != 1 {
		t.Fatalf("expected 1 download, got %d", hits)
	}
	if data, _ := os.ReadFile(dest); string(data) != "model-bytes" {
		t.Fatalf("content = %q", data)
	}
}

func TestEnsureHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "x.bin")

	_, err := Ensure(dest, srv.URL)

	if err == nil {
		t.Fatal("expected error")
	}
	if _, statErr := os.Stat(dest); statErr == nil {
		t.Fatal("partial file must not remain")
	}
}

func TestHuggingFaceURL(t *testing.T) {
	got := HuggingFaceURL("org/repo", "onnx/model.onnx")

	if got != "https://huggingface.co/org/repo/resolve/main/onnx/model.onnx" {
		t.Fatalf("got %q", got)
	}
}
