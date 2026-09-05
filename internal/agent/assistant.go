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

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/llm"
	"github.com/bryfur/ovi-voice-assistant/internal/mcp"
	"github.com/bryfur/ovi-voice-assistant/internal/memory"
)

// MaxTurns bounds the tool-calling loop per user input.
const MaxTurns = 10

// FailureMessage is spoken when the model call fails.
const FailureMessage = "Sorry, I could not process that."

// ChatStreamer is the model interface used by the agent loop.
type ChatStreamer interface {
	ChatStream(ctx context.Context, req llm.ChatRequest, onDelta func(string)) (*llm.StreamResult, error)
}

// SubAgentConfig defines a sub-agent exposed to the main agent as a tool.
type SubAgentConfig struct {
	Name         string             `json:"name"`
	Description  string             `json:"description"`
	Instructions string             `json:"instructions"`
	MCPServers   []mcp.ServerConfig `json:"mcp_servers"`
}

// ParseSubAgents parses a JSON array of SubAgentConfig, or "@path".
func ParseSubAgents(raw string) ([]SubAgentConfig, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var data []byte
	if strings.HasPrefix(raw, "@") {
		b, err := os.ReadFile(config.ExpandUser(raw[1:]))
		if err != nil {
			return nil, fmt.Errorf("read agents config: %w", err)
		}
		data = b
	} else {
		data = []byte(raw)
	}
	var agents []SubAgentConfig
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

// subAgent is a nested agent with its own instructions and MCP tools.
type subAgent struct {
	cfg     SubAgentConfig
	servers []*mcp.Client
}

// Assistant is a conversational agent with token streaming, built-in tools,
// MCP servers and sub-agents.
type Assistant struct {
	settings *config.Settings
	model    ChatStreamer

	tools      []Tool
	mcpClients []*mcp.Client
	subAgents  []*subAgent

	mu      sync.Mutex
	history []llm.Message
	loaded  bool

	// NewModel constructs the model client; tests may override it.
	NewModel func(settings *config.Settings) ChatStreamer
}

// New creates an unloaded assistant.
func New(settings *config.Settings) *Assistant {
	return &Assistant{
		settings: settings,
		NewModel: func(s *config.Settings) ChatStreamer {
			return llm.New(s.LLM.BaseURL, s.LLM.APIKey)
		},
	}
}

// Load parses the MCP/sub-agent configuration and builds the model client.
func (a *Assistant) Load() error {
	slog.Info("Initializing agent", "model", a.settings.LLM.Model)
	if a.settings.LLM.BaseURL != "" {
		slog.Info("  Base URL", "url", a.settings.LLM.BaseURL)
	}
	servers, err := mcp.ParseServers(a.settings.LLM.MCPServers)
	if err != nil {
		return err
	}
	a.mcpClients = nil
	for _, s := range servers {
		a.mcpClients = append(a.mcpClients, mcp.NewClient(s))
	}
	if len(a.mcpClients) > 0 {
		names := make([]string, len(a.mcpClients))
		for i, c := range a.mcpClients {
			names[i] = c.Name()
		}
		slog.Info("  MCP servers", "names", names)
	}

	a.model = a.NewModel(a.settings)

	subs, err := ParseSubAgents(a.settings.LLM.Agents)
	if err != nil {
		return err
	}
	a.subAgents = nil
	a.tools = BuiltinTools()
	for _, cfg := range subs {
		sa := &subAgent{cfg: cfg}
		for _, s := range cfg.MCPServers {
			c := mcp.NewClient(s)
			sa.servers = append(sa.servers, c)
			a.mcpClients = append(a.mcpClients, c)
		}
		a.subAgents = append(a.subAgents, sa)
		a.tools = append(a.tools, a.subAgentTool(sa))
		slog.Info("  Sub-agent", "name", cfg.Name)
	}
	a.loaded = true
	slog.Info("Agent initialized")
	return nil
}

// Start launches MCP server connections.
func (a *Assistant) Start(ctx context.Context) error {
	for _, c := range a.mcpClients {
		if err := c.Start(ctx); err != nil {
			return err
		}
		slog.Info("MCP server started", "name", c.Name(), "tools", len(c.Tools()))
	}
	return nil
}

// Stop terminates MCP server connections.
func (a *Assistant) Stop(ctx context.Context) error {
	for _, c := range a.mcpClients {
		done := make(chan struct{})
		go func(c *mcp.Client) {
			defer close(done)
			if err := c.Close(); err != nil {
				slog.Error("Error stopping MCP server", "name", c.Name(), "err", err)
			}
		}(c)
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			slog.Debug("MCP server did not stop cleanly", "name", c.Name())
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// ResetHistory clears conversation history. Call at the start of a new
// wake word session.
func (a *Assistant) ResetHistory() {
	a.mu.Lock()
	a.history = nil
	a.mu.Unlock()
	slog.Debug("Conversation history reset")
}

// History returns a copy of the conversation history.
func (a *Assistant) History() []llm.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]llm.Message(nil), a.history...)
}

// RunText runs the agent and returns the full response.
func (a *Assistant) RunText(ctx context.Context, text string, actx *Context) (string, error) {
	var sb strings.Builder
	err := a.RunStreamed(ctx, text, actx, func(tok string) { sb.WriteString(tok) })
	return sb.String(), err
}

// RunStreamed runs the agent with session history, invoking onToken for
// each content token. Model failures are logged and reported to the user
// as FailureMessage rather than returned as errors; only context
// cancellation is returned.
func (a *Assistant) RunStreamed(ctx context.Context, text string, actx *Context, onToken func(string)) error {
	if !a.loaded {
		return errors.New("call Load() first")
	}
	slog.Debug("Agent processing", "text", truncate(text, 80))

	// Auto-recall: inject relevant memories into the prompt
	input := text
	if actx != nil && actx.Memory != nil {
		input = a.injectMemories(ctx, text, actx)
	}

	a.mu.Lock()
	history := append([]llm.Message(nil), a.history...)
	a.mu.Unlock()

	messages := make([]llm.Message, 0, len(history)+2)
	messages = append(messages, llm.Message{Role: "system", Content: a.settings.LLM.Instructions})
	messages = append(messages, history...)
	messages = append(messages, llm.Message{Role: "user", Content: input})

	tools := a.toolDefs()
	if err := a.loop(ctx, &messages, tools, a.allTools(), actx, onToken); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		slog.Error("Agent call failed", "err", err)
		if onToken != nil {
			onToken(FailureMessage)
		}
	}

	a.mu.Lock()
	a.history = messages[1:] // drop system prompt
	a.mu.Unlock()
	slog.Debug("Agent response complete")
	return nil
}

// loop runs the model/tool loop until a final answer is produced.
func (a *Assistant) loop(ctx context.Context, messages *[]llm.Message, defs []llm.ToolDef, lookup map[string]Handler, actx *Context, onToken func(string)) error {
	for turn := 0; turn < MaxTurns; turn++ {
		res, err := a.model.ChatStream(ctx, llm.ChatRequest{
			Model:    a.settings.LLM.Model,
			Messages: *messages,
			Tools:    defs,
		}, onToken)
		if err != nil {
			return err
		}
		*messages = append(*messages, res.AssistantMessage())
		if len(res.ToolCalls) == 0 {
			return nil
		}
		for _, call := range res.ToolCalls {
			result := a.executeTool(ctx, call, lookup, actx)
			*messages = append(*messages, llm.Message{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    result,
			})
		}
	}
	return fmt.Errorf("max turns (%d) exceeded", MaxTurns)
}

func (a *Assistant) executeTool(ctx context.Context, call llm.ToolCall, lookup map[string]Handler, actx *Context) string {
	name := call.Function.Name
	handler, ok := lookup[name]
	if !ok {
		slog.Warn("Model called unknown tool", "name", name)
		return fmt.Sprintf("Error: unknown tool '%s'", name)
	}
	args, err := ParseArgs(call.Function.Arguments)
	if err != nil {
		return "Error: " + err.Error()
	}
	slog.Debug("Tool call", "name", name, "args", truncate(call.Function.Arguments, 200))
	if actx == nil {
		actx = &Context{}
	}
	out, err := handler(ctx, actx, args)
	if err != nil {
		slog.Error("Tool failed", "name", name, "err", err)
		if out == "" {
			return "Error: " + err.Error()
		}
	}
	return out
}

// toolDefs returns the model-visible tool definitions: enabled builtins,
// sub-agents and MCP tools.
func (a *Assistant) toolDefs() []llm.ToolDef {
	var defs []llm.ToolDef
	seen := map[string]bool{}
	for _, t := range a.tools {
		if !t.Enabled || seen[t.Name] {
			continue
		}
		seen[t.Name] = true
		defs = append(defs, t.Def())
	}
	for _, c := range a.topLevelMCP() {
		for _, mt := range c.Tools() {
			if seen[mt.Name] {
				slog.Warn("Duplicate tool name ignored", "name", mt.Name, "server", c.Name())
				continue
			}
			seen[mt.Name] = true
			defs = append(defs, mcpToolDef(mt))
		}
	}
	return defs
}

// allTools maps tool names to handlers for the main agent.
func (a *Assistant) allTools() map[string]Handler {
	lookup := map[string]Handler{}
	for _, t := range a.tools {
		if _, dup := lookup[t.Name]; !dup {
			lookup[t.Name] = t.Handler
		}
	}
	for _, c := range a.topLevelMCP() {
		for _, mt := range c.Tools() {
			if _, dup := lookup[mt.Name]; dup {
				continue
			}
			lookup[mt.Name] = mcpHandler(c, mt.Name)
		}
	}
	return lookup
}

// topLevelMCP returns MCP clients attached to the main agent (not sub-agents).
func (a *Assistant) topLevelMCP() []*mcp.Client {
	sub := map[*mcp.Client]bool{}
	for _, sa := range a.subAgents {
		for _, c := range sa.servers {
			sub[c] = true
		}
	}
	var out []*mcp.Client
	for _, c := range a.mcpClients {
		if !sub[c] {
			out = append(out, c)
		}
	}
	return out
}

func mcpToolDef(t mcp.Tool) llm.ToolDef {
	params := t.InputSchema
	if params == nil {
		params = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return llm.ToolDef{Type: "function", Function: llm.FunctionDef{
		Name: t.Name, Description: t.Description, Parameters: params,
	}}
}

func mcpHandler(c *mcp.Client, name string) Handler {
	return func(ctx context.Context, _ *Context, args Args) (string, error) {
		return c.CallTool(ctx, name, args)
	}
}

// subAgentTool exposes a sub-agent as a tool taking a single "input".
func (a *Assistant) subAgentTool(sa *subAgent) Tool {
	return Tool{
		Name:        sa.cfg.Name,
		Description: sa.cfg.Description,
		Parameters: schema([]string{"input"}, map[string]any{
			"input": prop("string", "The request to hand to the "+sa.cfg.Name+" agent."),
		}),
		Enabled: true,
		Handler: func(ctx context.Context, actx *Context, args Args) (string, error) {
			return a.runSubAgent(ctx, sa, args.String("input", ""), actx)
		},
	}
}

// runSubAgent runs a nested, history-less loop with the sub-agent's tools.
func (a *Assistant) runSubAgent(ctx context.Context, sa *subAgent, input string, actx *Context) (string, error) {
	var defs []llm.ToolDef
	lookup := map[string]Handler{}
	for _, c := range sa.servers {
		for _, mt := range c.Tools() {
			if _, dup := lookup[mt.Name]; dup {
				continue
			}
			defs = append(defs, mcpToolDef(mt))
			lookup[mt.Name] = mcpHandler(c, mt.Name)
		}
	}
	messages := []llm.Message{
		{Role: "system", Content: sa.cfg.Instructions},
		{Role: "user", Content: input},
	}
	if err := a.loop(ctx, &messages, defs, lookup, actx, nil); err != nil {
		return "", err
	}
	last := messages[len(messages)-1]
	return last.Content, nil
}

// injectMemories recalls relevant memories and prepends them to the input.
func (a *Assistant) injectMemories(ctx context.Context, text string, actx *Context) string {
	result, err := actx.Memory.Recall(ctx, text, memory.BudgetMid, 256)
	if err != nil {
		slog.Error("Memory recall failed, proceeding without context", "err", err)
		return text
	}
	if len(result.Results) == 0 {
		return text
	}
	var sb strings.Builder
	for _, f := range result.Results {
		sb.WriteString("- ")
		sb.WriteString(f.Text)
		sb.WriteString("\n")
	}
	slog.Debug("Injected memories", "count", len(result.Results))
	return "[Relevant memories]\n" + strings.TrimRight(sb.String(), "\n") + "\n\n[User]\n" + text
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
