package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/bryfur/ovi-voice-assistant/internal/agent/mcp"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

const (
	maxTurns       = 10 // tool-calling rounds per request
	failureMessage = "Sorry, I could not process that."
)

// LevelTrace logs every streamed model chunk; `ovi --verbose` enables it.
const LevelTrace = slog.LevelDebug - 4

type messages = []openai.ChatCompletionMessageParamUnion

// subAgent is a nested agent the main one can call as a tool, with its own
// instructions and MCP servers.
type subAgent struct {
	Name         string       `json:"name"`
	Description  string       `json:"description"`
	Instructions string       `json:"instructions"`
	Servers      []mcp.Server `json:"mcp_servers"`

	clients []*mcp.Client
}

// Assistant answers with tools and remembers the conversation.
type Assistant struct {
	cfg     config.LLMConfig
	client  openai.Client
	opts    []option.RequestOption // per-request extras, such as thinking off
	tools   []Tool                 // builtins and sub-agents
	clients []*mcp.Client
	subs    []*subAgent

	mu      sync.Mutex
	history messages
}

// New prepares an assistant; Load reads its configuration.
func New(cfg config.LLMConfig) *Assistant { return &Assistant{cfg: cfg} }

// Load builds the API client and parses the MCP and sub-agent configs.
func (a *Assistant) Load() error {
	opts := []option.RequestOption{option.WithMaxRetries(1)}
	if a.cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(a.cfg.BaseURL))
	}
	if a.cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(a.cfg.APIKey))
	}
	a.client = openai.NewClient(opts...)
	a.opts = nil
	if !a.cfg.Reasoning && a.cfg.BaseURL != "" && !strings.Contains(a.cfg.BaseURL, "api.openai.com") {
		// Local servers often ignore reasoning_effort, so also send the
		// llama.cpp / vLLM / LM Studio and ollama spellings.
		a.opts = append(a.opts,
			option.WithJSONSet("chat_template_kwargs", map[string]any{"enable_thinking": false}),
			option.WithJSONSet("think", false))
	}

	var servers []mcp.Server
	if err := parseJSON(a.cfg.MCPServers, &servers); err != nil {
		return fmt.Errorf("mcp_servers: %w", err)
	}
	a.clients = nil
	for _, s := range servers {
		if s.Command == "" {
			return fmt.Errorf("mcp_servers: %s: missing command", s)
		}
		a.clients = append(a.clients, mcp.NewClient(s))
	}
	if err := parseJSON(a.cfg.Agents, &a.subs); err != nil {
		return fmt.Errorf("agents: %w", err)
	}
	a.tools = builtinTools()
	for _, sub := range a.subs {
		if sub.Name == "" {
			return errors.New("agents: every sub-agent needs a name")
		}
		sub.clients = nil
		for _, s := range sub.Servers {
			sub.clients = append(sub.clients, mcp.NewClient(s))
		}
		a.tools = append(a.tools, a.subTool(sub))
	}
	slog.Info("Agent ready", "model", a.cfg.Model, "mcp", len(a.clients), "sub_agents", len(a.subs))
	return nil
}

// parseJSON decodes inline JSON or the file named by "@path"; empty input
// leaves v untouched.
func parseJSON(raw string, v any) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	data := []byte(raw)
	if path, ok := strings.CutPrefix(raw, "@"); ok {
		var err error
		if data, err = os.ReadFile(config.ExpandUser(path)); err != nil {
			return err
		}
	}
	return json.Unmarshal(data, v)
}

func (a *Assistant) allClients() []*mcp.Client {
	all := append([]*mcp.Client(nil), a.clients...)
	for _, sub := range a.subs {
		all = append(all, sub.clients...)
	}
	return all
}

// Start launches the MCP servers.
func (a *Assistant) Start(ctx context.Context) error {
	for _, c := range a.allClients() {
		if err := c.Start(ctx); err != nil {
			return err
		}
		slog.Info("MCP server started", "name", c.String(), "tools", len(c.Tools()))
	}
	return nil
}

// Stop ends the MCP servers.
func (a *Assistant) Stop() {
	for _, c := range a.allClients() {
		_ = c.Close()
	}
}

// Reset forgets the conversation; each wake word starts a new one.
func (a *Assistant) Reset() {
	a.mu.Lock()
	a.history = nil
	a.mu.Unlock()
}

// Ask runs text through the agent and returns the whole answer.
func (a *Assistant) Ask(ctx context.Context, text string, env *Env) (string, error) {
	var sb strings.Builder
	err := a.Run(ctx, text, env, func(tok string) { sb.WriteString(tok) })
	return sb.String(), err
}

// Run answers text within the conversation, handing each token to emit as
// it streams. A failing model is reported by speaking failureMessage; only
// a cancelled ctx is an error.
func (a *Assistant) Run(ctx context.Context, text string, env *Env, emit func(string)) error {
	a.mu.Lock()
	msgs := append(messages{openai.SystemMessage(a.cfg.Instructions)}, a.history...)
	a.mu.Unlock()
	msgs = append(msgs, openai.UserMessage(text))

	err := a.complete(ctx, &msgs, merge(a.tools, mcpTools(a.clients)), env, emit)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		slog.Error("Agent failed", "err", err)
		if emit != nil {
			emit(failureMessage)
		}
	}
	a.mu.Lock()
	a.history = msgs[1:] // without the system prompt
	a.mu.Unlock()
	return nil
}

// complete streams completions and runs tool calls until the model
// answers without one.
func (a *Assistant) complete(ctx context.Context, msgs *messages, tools []Tool, env *Env, emit func(string)) error {
	defs := make([]openai.ChatCompletionToolUnionParam, len(tools))
	for i, t := range tools {
		defs[i] = t.Def()
	}
	for range maxTurns {
		params := openai.ChatCompletionNewParams{Model: shared.ChatModel(a.cfg.Model), Messages: *msgs}
		if len(defs) > 0 {
			params.Tools = defs
		}
		if !a.cfg.Reasoning {
			params.ReasoningEffort = shared.ReasoningEffortNone
		}
		slog.Debug("LLM request", "model", a.cfg.Model, "messages", len(*msgs), "tools", len(defs))
		msg, err := a.stream(ctx, params, emit)
		if err != nil {
			return err
		}
		*msgs = append(*msgs, msg.ToParam())
		if len(msg.ToolCalls) == 0 {
			return nil
		}
		for _, call := range msg.ToolCalls {
			result := a.call(ctx, tools, env, call.Function.Name, call.Function.Arguments)
			*msgs = append(*msgs, openai.ToolMessage(result, call.ID))
		}
	}
	return fmt.Errorf("max turns (%d) exceeded", maxTurns)
}

// stream makes one model call, emitting content as it arrives.
func (a *Assistant) stream(ctx context.Context, params openai.ChatCompletionNewParams, emit func(string)) (openai.ChatCompletionMessage, error) {
	start := time.Now()
	trace := slog.Default().Enabled(ctx, LevelTrace)
	var acc openai.ChatCompletionAccumulator
	var first time.Duration
	s := a.client.Chat.Completions.NewStreaming(ctx, params, a.opts...)
	for s.Next() {
		chunk := s.Current()
		acc.AddChunk(chunk)
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			if first == 0 {
				first = time.Since(start)
			}
			if emit != nil {
				emit(delta.Content)
			}
		}
		if trace { // the raw delta shows fields the SDK does not model, such as reasoning_content
			slog.Log(ctx, LevelTrace, "LLM chunk", "t", time.Since(start).Round(time.Millisecond), "delta", truncate(delta.RawJSON(), 300))
		}
	}
	if err := s.Err(); err != nil {
		return openai.ChatCompletionMessage{}, err
	}
	if len(acc.Choices) == 0 {
		return openai.ChatCompletionMessage{}, errors.New("empty completion")
	}
	msg := acc.Choices[0].Message
	slog.Debug("LLM response", "first_content", first.Round(time.Millisecond), "total", time.Since(start).Round(time.Millisecond),
		"finish", acc.Choices[0].FinishReason, "tool_calls", len(msg.ToolCalls), "content", truncate(msg.Content, 200))
	return msg, nil
}

// call runs one tool call and returns what the model should see.
func (a *Assistant) call(ctx context.Context, tools []Tool, env *Env, name, rawArgs string) string {
	var tool *Tool
	for i := range tools {
		if tools[i].Name == name {
			tool = &tools[i]
		}
	}
	if tool == nil {
		return "Error: unknown tool '" + name + "'"
	}
	args, err := parseArgs(rawArgs)
	if err != nil {
		return "Error: " + err.Error()
	}
	slog.Debug("Tool call", "name", name, "args", rawArgs)
	if env == nil {
		env = &Env{}
	}
	out, err := tool.Run(ctx, env, args)
	if err != nil {
		slog.Error("Tool failed", "name", name, "err", err)
		if out == "" {
			return "Error: " + err.Error()
		}
	}
	return out
}

// merge concatenates tool lists, keeping the first tool of each name.
func merge(lists ...[]Tool) []Tool {
	var out []Tool
	seen := map[string]bool{}
	for _, list := range lists {
		for _, t := range list {
			if seen[t.Name] {
				slog.Warn("Duplicate tool name ignored", "name", t.Name)
				continue
			}
			seen[t.Name] = true
			out = append(out, t)
		}
	}
	return out
}

// mcpTools wraps the servers' current tools.
func mcpTools(clients []*mcp.Client) []Tool {
	var tools []Tool
	for _, c := range clients {
		for _, t := range c.Tools() {
			tools = append(tools, Tool{Name: t.Name, Description: t.Description, Parameters: t.InputSchema,
				Run: func(ctx context.Context, _ *Env, args Args) (string, error) { return c.Call(ctx, t.Name, args) }})
		}
	}
	return tools
}

// subTool exposes a sub-agent as a tool taking a single "input".
func (a *Assistant) subTool(sub *subAgent) Tool {
	return Tool{
		Name:        sub.Name,
		Description: sub.Description,
		Parameters:  schema([]string{"input"}, map[string]any{"input": prop("string", "The request for the "+sub.Name+" agent.")}),
		Run: func(ctx context.Context, env *Env, args Args) (string, error) {
			msgs := messages{openai.SystemMessage(sub.Instructions), openai.UserMessage(args.String("input", ""))}
			if err := a.complete(ctx, &msgs, mcpTools(sub.clients), env, nil); err != nil {
				return "", err
			}
			if last := msgs[len(msgs)-1]; last.OfAssistant != nil {
				return last.OfAssistant.Content.OfString.Value, nil
			}
			return "", nil
		},
	}
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}
