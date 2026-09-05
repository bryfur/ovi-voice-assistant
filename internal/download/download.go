// Package download fetches model files into the local cache.
package download

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// HTTPClient is used for all downloads; tests may replace it.
var HTTPClient = &http.Client{Timeout: 0}

// Ensure downloads url to dest if dest does not already exist (or is empty).
// It returns dest.
func Ensure(dest, url string) (string, error) {
	if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
		return dest, nil
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", err
	}
	slog.Info("Downloading", "file", filepath.Base(dest), "url", url)
	start := time.Now()
	resp, err := HTTPClient.Get(url)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %s", url, resp.Status)
	}
	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	n, err := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if err != nil {
		os.Remove(tmp)
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	if closeErr != nil {
		os.Remove(tmp)
		return "", closeErr
	}
	if err := os.Rename(tmp, dest); err != nil {
		return "", err
	}
	slog.Info("Downloaded", "file", filepath.Base(dest), "mb", n/1_000_000,
		"secs", int(time.Since(start).Seconds()))
	return dest, nil
}

// HuggingFaceURL builds a resolve URL for a file in a HF repo.
func HuggingFaceURL(repo, path string) string {
	return "https://huggingface.co/" + repo + "/resolve/main/" + path
}
