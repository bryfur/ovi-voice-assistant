package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewDefaults(t *testing.T) {
	c := New("", "")

	if c.BaseURL != DefaultBaseURL || c.APIKey != "not-set" {
		t.Fatalf("got %+v", c)
	}
}

func TestNewTrimsTrailingSlash(t *testing.T) {
	c := New("http://localhost:11434/v1/", "k")

	if c.BaseURL != "http://localhost:11434/v1" {
		t.Fatalf("got %q", c.BaseURL)
	}
}

func TestChatNonStreaming(t *testing.T) {
	var gotAuth string
	var gotReq ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewDecoder(r.Body).Decode(&gotReq)
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Role: "assistant", Content: "hi"}}}})
	}))
	defer srv.Close()
	c := New(srv.URL, "secret")

	resp, err := c.Chat(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "x"}}})

	if err != nil || resp.Text() != "hi" {
		t.Fatalf("got %v, %v", resp, err)
	}
	if gotAuth != "Bearer secret" || gotReq.Stream || gotReq.Model != "m" {
		t.Fatalf("request = %+v auth=%q", gotReq, gotAuth)
	}
}

func TestChatAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := New(srv.URL, "k")

	_, err := c.Chat(context.Background(), ChatRequest{})

	var apiErr *APIError
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 error, got %v", err)
	}
	if ok := errorsAs(err, &apiErr); !ok || apiErr.Status != 401 {
		t.Fatalf("expected APIError, got %T", err)
	}
}

func errorsAs(err error, target **APIError) bool {
	e, ok := err.(*APIError)
	if ok {
		*target = e
	}
	return ok
}

func sse(lines ...string) string {
	var sb strings.Builder
	for _, l := range lines {
		sb.WriteString("data: " + l + "\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

func TestChatStreamAssemblesContentAndToolCalls(t *testing.T) {
	body := sse(
		`{"choices":[{"delta":{"content":"Hel"}}]}`,
		`{"choices":[{"delta":{"content":"lo"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_","arguments":"{\"a\""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"time","arguments":":1}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"other","arguments":"{}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if !req.Stream {
			t.Error("expected stream=true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, body)
	}))
	defer srv.Close()
	c := New(srv.URL, "k")
	var deltas []string

	res, err := c.ChatStream(context.Background(), ChatRequest{Model: "m"}, func(d string) { deltas = append(deltas, d) })

	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Hello" || strings.Join(deltas, "|") != "Hel|lo" {
		t.Fatalf("content=%q deltas=%v", res.Content, deltas)
	}
	if len(res.ToolCalls) != 2 || res.ToolCalls[0].ID != "call_1" ||
		res.ToolCalls[0].Function.Name != "get_time" || res.ToolCalls[0].Function.Arguments != `{"a":1}` ||
		res.ToolCalls[1].Function.Name != "other" {
		t.Fatalf("tool calls = %+v", res.ToolCalls)
	}
	if res.FinishReason != "tool_calls" {
		t.Fatalf("finish = %q", res.FinishReason)
	}
	msg := res.AssistantMessage()
	if msg.Role != "assistant" || len(msg.ToolCalls) != 2 {
		t.Fatalf("assistant message = %+v", msg)
	}
}

func TestChatStreamErrorChunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, sse(`{"error":{"message":"boom"}}`))
	}))
	defer srv.Close()
	c := New(srv.URL, "k")

	_, err := c.ChatStream(context.Background(), ChatRequest{}, nil)

	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("got %v", err)
	}
}

func TestChatStreamAssignsMissingToolCallIDs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, sse(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]}}]}`))
	}))
	defer srv.Close()
	c := New(srv.URL, "k")

	res, err := c.ChatStream(context.Background(), ChatRequest{}, nil)

	if err != nil || len(res.ToolCalls) != 1 || res.ToolCalls[0].ID == "" {
		t.Fatalf("got %+v, %v", res, err)
	}
}

func TestTranscribeMultipart(t *testing.T) {
	var gotModel, gotLang, gotFile string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		r.ParseMultipartForm(1 << 20)
		gotModel = r.FormValue("model")
		gotLang = r.FormValue("language")
		f, _, _ := r.FormFile("file")
		b, _ := io.ReadAll(f)
		gotFile = string(b)
		io.WriteString(w, `{"text":"  hello world \n"}`)
	}))
	defer srv.Close()
	c := New(srv.URL, "k")

	text, err := c.Transcribe(context.Background(), []byte("RIFF"), "whisper-1", "en")

	if err != nil || text != "hello world" {
		t.Fatalf("got %q, %v", text, err)
	}
	if gotModel != "whisper-1" || gotLang != "en" || gotFile != "RIFF" {
		t.Fatalf("form = %q %q %q", gotModel, gotLang, gotFile)
	}
}
