package agent

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// The SDK must hand tokens over as they arrive, not after the stream ends.
func TestTokensArriveIncrementally(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for i := range 10 {
			fmt.Fprintf(w, `data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"w%d "},"finish_reason":null}]}`+"\n\n", i)
			fl.Flush()
			time.Sleep(100 * time.Millisecond)
		}
		fmt.Fprint(w, `data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	cfg := config.Default().LLM
	cfg.BaseURL, cfg.APIKey = srv.URL, "x"
	a := New(cfg)
	a.Load()
	start := time.Now()
	var first, last time.Duration

	a.RunStreamed(context.Background(), "hi", nil, func(string) {
		if first == 0 {
			first = time.Since(start)
		}
		last = time.Since(start)
	})

	t.Logf("first token at %v, last at %v", first.Round(time.Millisecond), last.Round(time.Millisecond))
	if first > 300*time.Millisecond || last-first < 700*time.Millisecond {
		t.Fatalf("tokens were buffered: first=%v last=%v", first, last)
	}
}
