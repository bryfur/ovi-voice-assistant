package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

var ctx = context.Background()

func newAssistant(t *testing.T, f *fakeOpenAI, mutate func(*config.LLMConfig)) *Assistant {
	t.Helper()
	cfg := config.Default().LLM
	cfg.BaseURL, cfg.APIKey = f.URL, "test"
	if mutate != nil {
		mutate(&cfg)
	}
	a := New(cfg)
	if err := a.Load(); err != nil {
		t.Fatal(err)
	}
	return a
}

func TestRunStreamsTokensAndKeepsHistory(t *testing.T) {
	f := newFakeOpenAI(t, turn{text: "Hello there"}, turn{text: "Again"}, turn{text: "Fresh"})
	a := newAssistant(t, f, nil)
	var tokens []string

	err := a.Run(ctx, "hi", nil, func(s string) { tokens = append(tokens, s) })
	a.Ask(ctx, "again", nil)
	a.Reset()
	a.Ask(ctx, "fresh", nil)

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

func TestRunExecutesToolCalls(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"c1", "calculate", `{"expression":"2+2"}`}}},
		turn{text: "It is 4"},
	)
	a := newAssistant(t, f, nil)

	out, err := a.Ask(ctx, "what is 2+2", &Env{})

	if err != nil || out != "It is 4" {
		t.Fatalf("got %q, %v", out, err)
	}
	last := f.Reqs[1].last()
	if last["role"] != "tool" || last["tool_call_id"] != "c1" || last["content"] != "4" {
		t.Fatalf("tool message = %v", last)
	}
	if assistant := f.Reqs[1].Messages[2]; assistant["role"] != "assistant" || assistant["tool_calls"] == nil {
		t.Fatalf("assistant tool-call message = %v", assistant)
	}
}

func TestRunReportsUnknownToolsAndBadArgs(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"1", "nope", `{}`}, {"2", "calculate", `{bad`}}},
		turn{text: "ok"},
	)
	a := newAssistant(t, f, nil)

	out, _ := a.Ask(ctx, "x", nil)

	msgs := f.Reqs[1].Messages
	unknown, bad := msgs[len(msgs)-2]["content"].(string), msgs[len(msgs)-1]["content"].(string)
	if out != "ok" || !strings.HasPrefix(unknown, "Error: unknown tool") || !strings.HasPrefix(bad, "Error:") {
		t.Fatalf("out=%q msgs=%v", out, msgs)
	}
}

func TestModelFailureIsSpokenAndCancellationReturned(t *testing.T) {
	a := newAssistant(t, newFakeOpenAI(t, turn{fail: true}), nil)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	out, err := a.Ask(ctx, "x", nil)
	cerr := a.Run(cancelled, "x", nil, nil)

	if err != nil || out != failureMessage || !errors.Is(cerr, context.Canceled) {
		t.Fatalf("got %q, %v, %v", out, err, cerr)
	}
}

func TestMaxTurnsExceeded(t *testing.T) {
	turns := slices.Repeat([]turn{{calls: []fakeCall{{"1", "flip_coin", "{}"}}}}, maxTurns+2)
	a := newAssistant(t, newFakeOpenAI(t, turns...), nil)

	out, _ := a.Ask(ctx, "x", &Env{})

	if out != failureMessage {
		t.Fatalf("got %q", out)
	}
}

func TestSubAgentExposedAsTool(t *testing.T) {
	f := newFakeOpenAI(t,
		turn{calls: []fakeCall{{"1", "web_search", `{"input":"news"}`}}},
		turn{text: "sub-agent answer"}, // the nested conversation
		turn{text: "final"},
	)
	a := newAssistant(t, f, func(c *config.LLMConfig) {
		c.Agents = `[{"name":"web_search","description":"Search the web","instructions":"You browse."}]`
	})

	out, err := a.Ask(ctx, "news?", &Env{})

	if err != nil || out != "final" || !slices.Contains(f.Reqs[0].toolNames(), "web_search") {
		t.Fatalf("got %q, %v, tools %v", out, err, f.Reqs[0].toolNames())
	}
	if nested := f.Reqs[1]; nested.content(0) != "You browse." || nested.content(1) != "news" || len(nested.Tools) != 0 {
		t.Fatalf("nested request = %+v", nested)
	}
	if f.Reqs[2].last()["content"] != "sub-agent answer" {
		t.Fatalf("tool result = %v", f.Reqs[2].last())
	}
}

func TestReasoningOffSendsDisableFields(t *testing.T) {
	f := newFakeOpenAI(t, turn{text: "ok"}, turn{text: "ok"})
	off := newAssistant(t, f, func(c *config.LLMConfig) { c.Reasoning = false })
	on := newAssistant(t, f, nil)

	off.Ask(ctx, "x", nil)
	on.Ask(ctx, "x", nil)

	got := f.Reqs[0]
	if got.ReasoningEffort != "none" || got.TemplateKwargs["enable_thinking"] != false || got.Think == nil || *got.Think {
		t.Fatalf("reasoning-off request = %+v", got)
	}
	if def := f.Reqs[1]; def.ReasoningEffort != "" || def.TemplateKwargs != nil || def.Think != nil {
		t.Fatalf("default request must not mention reasoning: %+v", def)
	}
}

func TestLoadParsesConfigFilesAndRejectsBadOnes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agents.json")
	os.WriteFile(path, []byte(`[{"name":"a","mcp_servers":[{"command":"npx","args":["x"]}]}]`), 0o644)
	cfg := config.Default().LLM

	cfg.Agents = "@" + path
	a := New(cfg)
	err := a.Load()

	if err != nil || len(a.subs) != 1 || len(a.subs[0].clients) != 1 {
		t.Fatalf("subs = %+v, %v", a.subs, err)
	}
	for _, bad := range []config.LLMConfig{{Agents: `[{"description":"no name"}]`}, {MCPServers: "{not json"}, {MCPServers: `[{"args":["x"]}]`}} {
		if New(bad).Load() == nil {
			t.Fatalf("expected error for %+v", bad)
		}
	}
}

func TestStartStopWithoutServers(t *testing.T) {
	a := newAssistant(t, newFakeOpenAI(t), nil)

	if err := a.Start(ctx); err != nil {
		t.Fatal(err)
	}
	a.Stop()
}
