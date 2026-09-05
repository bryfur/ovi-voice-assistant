package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/agent/mcp"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// maxTurns bounds the tool-calling loop per user input.
const maxTurns = 10

// failureMessage is spoken when the model call fails.
const failureMessage = "Sorry, I could not process that."

// LevelTrace logs every streamed LLM chunk; enabled by `ovi --verbose`.
const LevelTrace = slog.LevelDebug - 4

// SubAgent is a nested agent exposed to the main agent as a tool.
type SubAgent struct {
	Name         string             `json:"name"`
	Description  string             `json:"description"`
	Instructions string             `json:"instructions"`
	MCPServers   []mcp.ServerConfig `json:"mcp_servers"`

	servers []*mcp.Client
}

// parseSubAgents parses a JSON array of sub-agents, or "@path" to a file.
func parseSubAgents(raw string) ([]*SubAgent, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	data := []byte(raw)
	if strings.HasPrefix(raw, "@") {
		b, err := os.ReadFile(config.ExpandUser(raw[1:]))
		if err != nil {
			return nil, fmt.Errorf("read agents config: %w", err)
		}
		data = b
	}
	var agents []*SubAgent
	if err := json.Unmarshal(data, &agents); err != nil {
		return nil, fmt.Errorf("parse agents config: %w", err)
	}
	for i, a := range agents {
		if a.Name == "" {
			return nil, fmt.Errorf("agent %d: missing name", i)
		}
	}
	return agents, nil
}

type messages = []openai.ChatCompletionMessageParamUnion

// Assistant runs the model with tools and keeps per-session history.
type Assistant struct {
	cfg     config.LLMConfig
	client  openai.Client
	reqOpts []option.RequestOption // per-request extras, e.g. thinking off

	tools   []Tool // builtins + sub-agents
	mcp     []*mcp.Client
	subs    []*SubAgent
	mu      sync.Mutex
	history messages
}

// New creates an unloaded assistant.
func New(cfg config.LLMConfig) *Assistant { return &Assistant{cfg: cfg} }

// Load parses the MCP and sub-agent configuration and builds the client.
func (a *Assistant) Load() error {
	opts := []option.RequestOption{option.WithMaxRetries(1)}
	if a.cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(a.cfg.BaseURL))
	}
	if a.cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(a.cfg.APIKey))
	}
	a.client = openai.NewClient(opts...)
	a.reqOpts = nil
	if !a.cfg.Reasoning && !strings.Contains(a.cfg.BaseURL, "api.openai.com") && a.cfg.BaseURL != "" {
		// Local OpenAI-compatible servers: the standard field alone is often
		// ignored, so also send the llama.cpp/vLLM/LM Studio and ollama forms.
		a.reqOpts = append(a.reqOpts,
			option.WithJSONSet("chat_template_kwargs", map[string]any{"enable_thinking": false}),
			option.WithJSONSet("think", false))
	}

	servers, err := mcp.ParseServers(a.cfg.MCPServers)
	if err != nil {
		return err
	}
	a.mcp = nil
	for _, s := range servers {
		a.mcp = append(a.mcp, mcp.NewClient(s))
	}
	if a.subs, err = parseSubAgents(a.cfg.Agents); err != nil {
		return err
	}
	a.tools = builtinTools()
	for _, sub := range a.subs {
		for _, s := range sub.MCPServers {
			sub.servers = append(sub.servers, mcp.NewClient(s))
		}
		a.tools = append(a.tools, a.subAgentTool(sub))
	}
	slog.Info("Agent ready", "model", a.cfg.Model, "mcp", len(a.mcp), "sub_agents", len(a.subs))
	return nil
}

func (a *Assistant) allMCP() []*mcp.Client {
	all := append([]*mcp.Client(nil), a.mcp...)
	for _, sub := range a.subs {
		all = append(all, sub.servers...)
	}
	return all
}

// Start launches MCP servers.
func (a *Assistant) Start(ctx context.Context) error {
	for _, c := range a.allMCP() {
		if err := c.Start(ctx); err != nil {
			return err
		}
		slog.Info("MCP server started", "name", c.Name(), "tools", len(c.Tools()))
	}
	return nil
}

// Stop terminates MCP servers.
func (a *Assistant) Stop(context.Context) error {
	for _, c := range a.allMCP() {
		_ = c.Close()
	}
	return nil
}

// ResetHistory starts a fresh conversation (called on each wake word).
func (a *Assistant) ResetHistory() {
	a.mu.Lock()
	a.history = nil
	a.mu.Unlock()
}

// RunText runs the agent and returns the full response.
func (a *Assistant) RunText(ctx context.Context, text string, actx *Context) (string, error) {
	var sb strings.Builder
	err := a.RunStreamed(ctx, text, actx, func(tok string) { sb.WriteString(tok) })
	return sb.String(), err
}

// RunStreamed runs the agent with session history, invoking onToken for
// each content token. Model failures are spoken as failureMessage; only
// context cancellation is returned as an error.
func (a *Assistant) RunStreamed(ctx context.Context, text string, actx *Context, onToken func(string)) error {
	a.mu.Lock()
	msgs := append(messages{openai.SystemMessage(a.cfg.Instructions)}, a.history...)
	a.mu.Unlock()
	msgs = append(msgs, openai.UserMessage(text))

	defs, handlers := a.toolset(a.tools, a.mcp)
	err := a.loop(ctx, &msgs, defs, handlers, actx, onToken)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		slog.Error("Agent call failed", "err", err)
		if onToken != nil {
			onToken(failureMessage)
		}
	}
	a.mu.Lock()
	a.history = msgs[1:] // drop the system prompt
	a.mu.Unlock()
	return nil
}

// loop streams completions and executes tool calls until the model
// answers without calling a tool.
func (a *Assistant) loop(ctx context.Context, msgs *messages, defs []openai.ChatCompletionToolUnionParam,
	handlers map[string]Handler, actx *Context, onToken func(string)) error {
	for range maxTurns {
		params := openai.ChatCompletionNewParams{Model: shared.ChatModel(a.cfg.Model), Messages: *msgs}
		if len(defs) > 0 {
			params.Tools = defs
		}
		if !a.cfg.Reasoning {
			params.ReasoningEffort = shared.ReasoningEffortNone
		}
		slog.Debug("LLM request", "model", a.cfg.Model, "messages", len(*msgs), "tools", len(defs))
		start := time.Now()
		trace := slog.Default().Enabled(ctx, LevelTrace)
		stream := a.client.Chat.Completions.NewStreaming(ctx, params, a.reqOpts...)
		var acc openai.ChatCompletionAccumulator
		var chunks int
		var firstContent time.Duration
		for stream.Next() {
			chunk := stream.Current()
			chunks++
			acc.AddChunk(chunk)
			var delta string
			raw := chunk.RawJSON()
			if len(chunk.Choices) > 0 {
				delta = chunk.Choices[0].Delta.Content
				raw = chunk.Choices[0].Delta.RawJSON()
			}
			if delta != "" {
				if firstContent == 0 {
					firstContent = time.Since(start)
				}
				if onToken != nil {
					onToken(delta)
				}
			}
			if trace {
				// The raw delta shows fields the SDK does not model, such as
				// reasoning_content from thinking models.
				slog.Log(ctx, LevelTrace, "LLM chunk", "t", time.Since(start).Round(time.Millisecond), "delta", truncate(raw, 300))
			}
		}
		if err := stream.Err(); err != nil {
			return err
		}
		if len(acc.Choices) == 0 {
			return errors.New("empty completion")
		}
		msg := acc.Choices[0].Message
		slog.Debug("LLM response", "chunks", chunks, "first_content", firstContent.Round(time.Millisecond),
			"total", time.Since(start).Round(time.Millisecond), "finish", acc.Choices[0].FinishReason,
			"tool_calls", len(msg.ToolCalls), "content", truncate(msg.Content, 200))
		*msgs = append(*msgs, msg.ToParam())
		if len(msg.ToolCalls) == 0 {
			return nil
		}
		for _, call := range msg.ToolCalls {
			result := a.call(ctx, handlers, actx, call.Function.Name, call.Function.Arguments)
			*msgs = append(*msgs, openai.ToolMessage(result, call.ID))
		}
	}
	return fmt.Errorf("max turns (%d) exceeded", maxTurns)
}

func (a *Assistant) call(ctx context.Context, handlers map[string]Handler, actx *Context, name, rawArgs string) string {
	h, ok := handlers[name]
	if !ok {
		return "Error: unknown tool '" + name + "'"
	}
	args, err := parseArgs(rawArgs)
	if err != nil {
		return "Error: " + err.Error()
	}
	slog.Debug("Tool call", "name", name, "args", rawArgs)
	if actx == nil {
		actx = &Context{}
	}
	out, err := h(ctx, actx, args)
	if err != nil {
		slog.Error("Tool failed", "name", name, "err", err)
		if out == "" {
			return "Error: " + err.Error()
		}
	}
	return out
}

// toolset builds the model-visible definitions and handler map for a set
// of tools and MCP servers. First definition of a name wins.
func (a *Assistant) toolset(tools []Tool, servers []*mcp.Client) ([]openai.ChatCompletionToolUnionParam, map[string]Handler) {
	var defs []openai.ChatCompletionToolUnionParam
	handlers := map[string]Handler{}
	add := func(t Tool) {
		if _, dup := handlers[t.Name]; dup {
			slog.Warn("Duplicate tool name ignored", "name", t.Name)
			return
		}
		handlers[t.Name] = t.Handler
		defs = append(defs, t.Def())
	}
	for _, t := range tools {
		add(t)
	}
	for _, c := range servers {
		for _, mt := range c.Tools() {
			add(Tool{Name: mt.Name, Description: mt.Description, Parameters: mt.InputSchema,
				Handler: func(ctx context.Context, _ *Context, args Args) (string, error) {
					return c.CallTool(ctx, mt.Name, args)
				}})
		}
	}
	return defs, handlers
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

// subAgentTool exposes a sub-agent as a tool with a single "input".
func (a *Assistant) subAgentTool(sub *SubAgent) Tool {
	return Tool{
		Name:        sub.Name,
		Description: sub.Description,
		Parameters: schema([]string{"input"}, map[string]any{
			"input": prop("string", "The request for the "+sub.Name+" agent."),
		}),
		Handler: func(ctx context.Context, actx *Context, args Args) (string, error) {
			msgs := messages{openai.SystemMessage(sub.Instructions), openai.UserMessage(args.String("input", ""))}
			defs, handlers := a.toolset(nil, sub.servers)
			if err := a.loop(ctx, &msgs, defs, handlers, actx, nil); err != nil {
				return "", err
			}
			last := msgs[len(msgs)-1]
			if last.OfAssistant != nil {
				return last.OfAssistant.Content.OfString.Value, nil
			}
			return "", nil
		},
	}
}
