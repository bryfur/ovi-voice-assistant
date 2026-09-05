package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// turn is one scripted model reply: text, tool calls, or an HTTP error.
type turn struct {
	text  string
	calls []fakeCall
	fail  bool
}

type fakeCall struct{ id, name, args string }

// request is what the fake server saw for one completion call.
type request struct {
	Messages        []map[string]any `json:"messages"`
	Tools           []map[string]any `json:"tools"`
	Stream          bool             `json:"stream"`
	ReasoningEffort string           `json:"reasoning_effort"`
	TemplateKwargs  map[string]any   `json:"chat_template_kwargs"`
	Think           *bool            `json:"think"`
}

func (r request) role(i int) string    { return fmt.Sprint(r.Messages[i]["role"]) }
func (r request) content(i int) string { return fmt.Sprint(r.Messages[i]["content"]) }
func (r request) last() map[string]any { return r.Messages[len(r.Messages)-1] }
func (r request) toolNames() (names []string) {
	for _, t := range r.Tools {
		names = append(names, fmt.Sprint(t["function"].(map[string]any)["name"]))
	}
	return
}

// fakeOpenAI serves scripted streaming completions and records requests.
type fakeOpenAI struct {
	URL   string
	mu    sync.Mutex
	turns []turn
	Reqs  []request
}

func newFakeOpenAI(t *testing.T, turns ...turn) *fakeOpenAI {
	t.Helper()
	f := &fakeOpenAI{turns: turns}
	srv := httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(srv.Close)
	f.URL = srv.URL
	return f
}

func (f *fakeOpenAI) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var req request
	_ = json.Unmarshal(body, &req)
	f.mu.Lock()
	f.Reqs = append(f.Reqs, req)
	i := len(f.Reqs) - 1
	f.mu.Unlock()
	if i >= len(f.turns) || f.turns[i].fail {
		http.Error(w, `{"error":{"message":"boom"}}`, http.StatusInternalServerError)
		return
	}
	tn := f.turns[i]
	w.Header().Set("Content-Type", "text/event-stream")
	chunk := func(delta string, finish string) {
		fmt.Fprintf(w, `data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":%s,"finish_reason":%s}]}`+"\n\n", delta, finish)
	}
	for _, word := range strings.SplitAfter(tn.text, " ") {
		if word != "" {
			b, _ := json.Marshal(map[string]any{"role": "assistant", "content": word})
			chunk(string(b), "null")
		}
	}
	for idx, c := range tn.calls {
		b, _ := json.Marshal(map[string]any{"role": "assistant", "tool_calls": []map[string]any{{
			"index": idx, "id": c.id, "type": "function",
			"function": map[string]any{"name": c.name, "arguments": c.args},
		}}})
		chunk(string(b), "null")
	}
	if len(tn.calls) > 0 {
		chunk("{}", `"tool_calls"`)
	} else {
		chunk("{}", `"stop"`)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}
