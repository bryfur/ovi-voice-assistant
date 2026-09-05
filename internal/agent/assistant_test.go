package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func newAssistant(t *testing.T, f *fakeOpenAI, mutate func(*config.LLMConfig)) *Assistant {
	t.Helper()
	cfg := config.Default().LLM
	cfg.BaseURL = f.URL
	cfg.APIKey = "test"
	if mutate != nil {
		mutate(&cfg)
	}
	a := New(cfg)
	if err := a.Load(); err != nil {
		t.Fatal(err)
	}
	return a
}

func TestRunStreamedYieldsTokensAndKeepsHistory(t *testing.T) {
	f := newFakeOpenAI(t, turn{text: "Hello there"}, turn{text: "Again"}, turn{text: "Fresh"})
	a := newAssistant(t, f, nil)
	var tokens []string

	err := a.RunStreamed(context.Background(), "hi", nil, func(s string) { tokens = append(tokens, s) })
	a.RunText(context.Background(), "again", nil)
	a.ResetHistory()
	a.RunText(context.Background(), "fresh", nil)

	if err != nil || strings.Join(tokens, "") != "Hello there" {
		t.Fatalf("tokens=%v err=%v", tokens, err)
	}
	first := f.Reqs[0]
	if !first.Stream || first.role(0) != "system" || first.content(0) != config.DefaultInstructions || first.content(1) != "hi" || len(first.Tools) != 19 {
		t.Fatalf("first request = %+v", first)
	}
	if second := f.Reqs[1]; len(second.Messages) != 4 || second.content(2) != "Hello there" || second.content(3) != "again" {
		t.Fatalf("history not kept: %+v", second.Messages)
	}
	if third := f.Reqs[2]; len(third.Messages) != 2 || third.content(1) != "fresh" {
		t.Fatalf("history not reset: %+v", third.Messages)
	}
}

func TestRunStreamedExecutesToolCalls(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"c1", "calculate", `{"expression":"2+2"}`}}},
		turn{text: "It is 4"},
	)
	a := newAssistant(t, f, nil)

	out, err := a.RunText(context.Background(), "what is 2+2", &Context{})

	if err != nil || out != "It is 4" {
		t.Fatalf("got %q, %v", out, err)
	}
	last := f.Reqs[1].last()
	if last["role"] != "tool" || last["tool_call_id"] != "c1" || last["content"] != "4" {
		t.Fatalf("tool message = %v", last)
	}
	assistant := f.Reqs[1].Messages[2]
	if assistant["role"] != "assistant" || assistant["tool_calls"] == nil {
		t.Fatalf("assistant tool-call message = %v", assistant)
	}
}

func TestRunStreamedUnknownToolAndBadArgs(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"1", "nope", `{}`}, {"2", "calculate", `{bad`}}},
		turn{text: "ok"},
	)
	a := newAssistant(t, f, nil)

	out, _ := a.RunText(context.Background(), "x", nil)

	msgs := f.Reqs[1].Messages
	if out != "ok" || !strings.HasPrefix(str(msgs[len(msgs)-2]["content"]), "Error: unknown tool") ||
		!strings.HasPrefix(str(msgs[len(msgs)-1]["content"]), "Error:") {
		t.Fatalf("out=%q msgs=%v", out, msgs)
	}
}

func str(v any) string { return v.(string) }

func TestRunStreamedModelFailureSpeaksApology(t *testing.T) {
	f := newFakeOpenAI(t, turn{fail: true})
	a := newAssistant(t, f, nil)

	out, err := a.RunText(context.Background(), "x", nil)

	if err != nil || out != FailureMessage {
		t.Fatalf("got %q, %v", out, err)
	}
}

func TestRunStreamedCancelledContextReturnsError(t *testing.T) {
	f := newFakeOpenAI(t, turn{text: "x"})
	a := newAssistant(t, f, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := a.RunStreamed(ctx, "x", nil, nil)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestMaxTurnsExceeded(t *testing.T) {
	loop := turn{calls: []fakeCall{{"1", "flip_coin", "{}"}}}
	var turns []turn
	for range MaxTurns + 2 {
		turns = append(turns, loop)
	}
	a := newAssistant(t, newFakeOpenAI(t, turns...), nil)

	out, _ := a.RunText(context.Background(), "x", &Context{})

	if out != FailureMessage {
		t.Fatalf("got %q", out)
	}
}

func TestSubAgentExposedAsTool(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"1", "web_search", `{"input":"news"}`}}},
		turn{text: "sub-agent answer"}, // nested loop
		turn{text: "final"},
	)
	a := newAssistant(t, f, func(c *config.LLMConfig) {
		c.Agents = `[{"name":"web_search","description":"Search the web","instructions":"You browse."}]`
	})

	out, err := a.RunText(context.Background(), "news?", &Context{})

	if err != nil || out != "final" {
		t.Fatalf("got %q, %v", out, err)
	}
	if names := f.Reqs[0].toolNames(); !contains(names, "web_search") {
		t.Fatalf("sub-agent tool not offered: %v", names)
	}
	nested := f.Reqs[1]
	if nested.content(0) != "You browse." || nested.content(1) != "news" || len(nested.Tools) != 0 {
		t.Fatalf("nested request = %+v", nested)
	}
	if f.Reqs[2].last()["content"] != "sub-agent answer" {
		t.Fatalf("tool result = %v", f.Reqs[2].last())
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func TestParseSubAgentsFromFileAndValidation(t *testing.T) {
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
	cfg := config.Default().LLM
	cfg.MCPServers = "{not json"

	if err := New(cfg).Load(); err == nil {
		t.Fatal("expected error")
	}
}

func TestStartStopWithoutServers(t *testing.T) {
	a := newAssistant(t, newFakeOpenAI(t), nil)

	if err := a.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := a.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
}
