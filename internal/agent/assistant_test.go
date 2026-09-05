package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/llm"
	"github.com/bryfur/ovi-voice-assistant/internal/memory"
)

// scriptedModel returns canned responses in order and records requests.
type scriptedModel struct {
	responses []*llm.StreamResult
	errs      []error
	requests  []llm.ChatRequest
}

func (m *scriptedModel) ChatStream(_ context.Context, req llm.ChatRequest, onDelta func(string)) (*llm.StreamResult, error) {
	m.requests = append(m.requests, req)
	i := len(m.requests) - 1
	if i < len(m.errs) && m.errs[i] != nil {
		return nil, m.errs[i]
	}
	if i >= len(m.responses) {
		return &llm.StreamResult{Content: "done"}, nil
	}
	res := m.responses[i]
	if onDelta != nil && res.Content != "" {
		for _, word := range strings.SplitAfter(res.Content, " ") {
			onDelta(word)
		}
	}
	return res, nil
}

func newAssistant(t *testing.T, model ChatStreamer, mutate func(*config.Settings)) *Assistant {
	t.Helper()
	s := config.Default()
	if mutate != nil {
		mutate(s)
	}
	a := New(s)
	a.NewModel = func(*config.Settings) ChatStreamer { return model }
	if err := a.Load(); err != nil {
		t.Fatal(err)
	}
	return a
}

func TestRunStreamedBeforeLoad(t *testing.T) {
	a := New(config.Default())

	err := a.RunStreamed(context.Background(), "hi", nil, nil)

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRunStreamedYieldsTokensAndKeepsHistory(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{{Content: "Hello there"}}}
	a := newAssistant(t, model, nil)
	var tokens []string

	err := a.RunStreamed(context.Background(), "hi", nil, func(s string) { tokens = append(tokens, s) })

	if err != nil || strings.Join(tokens, "") != "Hello there" {
		t.Fatalf("tokens=%v err=%v", tokens, err)
	}
	req := model.requests[0]
	if req.Messages[0].Role != "system" || req.Messages[0].Content != config.DefaultInstructions || req.Messages[1].Content != "hi" {
		t.Fatalf("messages = %+v", req.Messages)
	}
	if len(req.Tools) != 19 { // 20 builtins minus disabled say
		t.Fatalf("tools = %d", len(req.Tools))
	}
	hist := a.History()
	if len(hist) != 2 || hist[1].Content != "Hello there" {
		t.Fatalf("history = %+v", hist)
	}
	a.ResetHistory()
	if len(a.History()) != 0 {
		t.Fatal("history not reset")
	}
}

func TestRunStreamedExecutesToolCalls(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{
		{ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "calculate", Arguments: `{"expression":"2+2"}`}}}},
		{Content: "It is 4"},
	}}
	a := newAssistant(t, model, nil)

	out, err := a.RunText(context.Background(), "what is 2+2", &Context{})

	if err != nil || out != "It is 4" {
		t.Fatalf("got %q, %v", out, err)
	}
	second := model.requests[1].Messages
	toolMsg := second[len(second)-1]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "c1" || toolMsg.Content != "4" {
		t.Fatalf("tool message = %+v", toolMsg)
	}
	if second[len(second)-2].Role != "assistant" || len(second[len(second)-2].ToolCalls) != 1 {
		t.Fatalf("assistant tool-call message missing: %+v", second[len(second)-2])
	}
}

func TestRunStreamedUnknownToolAndBadArgs(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{
		{ToolCalls: []llm.ToolCall{
			{ID: "1", Function: llm.FunctionCall{Name: "nope", Arguments: `{}`}},
			{ID: "2", Function: llm.FunctionCall{Name: "calculate", Arguments: `{bad`}},
		}},
		{Content: "ok"},
	}}
	a := newAssistant(t, model, nil)

	out, _ := a.RunText(context.Background(), "x", nil)

	msgs := model.requests[1].Messages
	if out != "ok" || !strings.HasPrefix(msgs[len(msgs)-2].Content, "Error: unknown tool") || !strings.HasPrefix(msgs[len(msgs)-1].Content, "Error:") {
		t.Fatalf("out=%q msgs=%+v", out, msgs)
	}
}

func TestRunStreamedModelFailureSpeaksApology(t *testing.T) {
	model := &scriptedModel{errs: []error{errors.New("boom")}}
	a := newAssistant(t, model, nil)

	out, err := a.RunText(context.Background(), "x", nil)

	if err != nil || out != FailureMessage {
		t.Fatalf("got %q, %v", out, err)
	}
}

func TestRunStreamedCancelledContextReturnsError(t *testing.T) {
	model := &scriptedModel{errs: []error{context.Canceled}}
	a := newAssistant(t, model, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := a.RunStreamed(ctx, "x", nil, nil)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestMaxTurnsExceeded(t *testing.T) {
	loop := &llm.StreamResult{ToolCalls: []llm.ToolCall{{ID: "1", Function: llm.FunctionCall{Name: "flip_coin", Arguments: "{}"}}}}
	var responses []*llm.StreamResult
	for i := 0; i < MaxTurns+2; i++ {
		responses = append(responses, loop)
	}
	a := newAssistant(t, &scriptedModel{responses: responses}, nil)

	out, _ := a.RunText(context.Background(), "x", &Context{})

	if out != FailureMessage {
		t.Fatalf("got %q", out)
	}
}

type fakeMemory struct {
	facts  []string
	query  string
	retain int
}

func (f *fakeMemory) Recall(_ context.Context, query string, _ memory.Budget, _ int) (memory.RecallResult, error) {
	f.query = query
	var res memory.RecallResult
	for i, t := range f.facts {
		res.Results = append(res.Results, memory.Fact{ID: string(rune('a' + i)), Text: t})
	}
	return res, nil
}

func (f *fakeMemory) Retain(context.Context, string, string) (memory.RetainResult, error) {
	f.retain++
	return memory.RetainResult{Success: true}, nil
}

func TestMemoryInjection(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{{Content: "ok"}}}
	a := newAssistant(t, model, nil)
	mem := &fakeMemory{facts: []string{"Alice likes pizza"}}

	a.RunText(context.Background(), "what does Alice like?", &Context{Memory: mem})

	user := model.requests[0].Messages[1].Content
	if user != "[Relevant memories]\n- Alice likes pizza\n\n[User]\nwhat does Alice like?" || mem.query != "what does Alice like?" {
		t.Fatalf("user message = %q", user)
	}
}

func TestMemoryInjectionNoFacts(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{{Content: "ok"}}}
	a := newAssistant(t, model, nil)

	a.RunText(context.Background(), "hi", &Context{Memory: &fakeMemory{}})

	if model.requests[0].Messages[1].Content != "hi" {
		t.Fatal("input should be unchanged without memories")
	}
}

func TestSubAgentExposedAsTool(t *testing.T) {
	model := &scriptedModel{responses: []*llm.StreamResult{
		{ToolCalls: []llm.ToolCall{{ID: "1", Function: llm.FunctionCall{Name: "web_search", Arguments: `{"input":"news"}`}}}},
		{Content: "sub-agent answer"}, // nested loop response
		{Content: "final"},
	}}
	a := newAssistant(t, model, func(s *config.Settings) {
		s.LLM.Agents = `[{"name":"web_search","description":"Search the web","instructions":"You browse."}]`
	})

	out, err := a.RunText(context.Background(), "news?", &Context{})

	if err != nil || out != "final" {
		t.Fatalf("got %q, %v", out, err)
	}
	names := map[string]bool{}
	for _, d := range model.requests[0].Tools {
		names[d.Function.Name] = true
	}
	if !names["web_search"] {
		t.Fatal("sub-agent tool not offered")
	}
	nested := model.requests[1].Messages
	if nested[0].Content != "You browse." || nested[1].Content != "news" {
		t.Fatalf("nested messages = %+v", nested)
	}
	final := model.requests[2].Messages
	if final[len(final)-1].Content != "sub-agent answer" {
		t.Fatalf("tool result = %+v", final[len(final)-1])
	}
}

func TestParseSubAgentsFromFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agents.json")
	os.WriteFile(path, []byte(`[{"name":"a","mcp_servers":[{"command":"npx","args":["x"]}]}]`), 0o644)

	agents, err := ParseSubAgents("@" + path)

	if err != nil || len(agents) != 1 || len(agents[0].MCPServers) != 1 {
		t.Fatalf("got %+v, %v", agents, err)
	}
	if _, err := ParseSubAgents(`[{"description":"no name"}]`); err == nil {
		t.Fatal("expected error for missing name")
	}
}

func TestLoadRejectsBadMCPConfig(t *testing.T) {
	s := config.Default()
	s.LLM.MCPServers = "{not json"
	a := New(s)
	a.NewModel = func(*config.Settings) ChatStreamer { return &scriptedModel{} }

	if err := a.Load(); err == nil {
		t.Fatal("expected error")
	}
}

func TestStartStopWithoutServers(t *testing.T) {
	a := newAssistant(t, &scriptedModel{}, nil)

	if err := a.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := a.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
}
