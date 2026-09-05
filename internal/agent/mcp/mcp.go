// Package mcp is a minimal Model Context Protocol client over stdio:
// newline-delimited JSON-RPC 2.0 to a subprocess with initialize,
// tools/list and tools/call.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ProtocolVersion is the MCP revision we advertise.
const ProtocolVersion = "2025-06-18"

// ServerConfig describes how to launch an MCP server.
type ServerConfig struct {
	Name    string            `json:"name,omitempty"`
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// DisplayName returns the configured name or a name derived from the command.
func (c ServerConfig) DisplayName() string {
	if c.Name != "" {
		return c.Name
	}
	if len(c.Args) > 0 {
		return c.Args[len(c.Args)-1]
	}
	return c.Command
}

// ParseServers parses a JSON array of ServerConfig, or "@path" to a file.
func ParseServers(raw string) ([]ServerConfig, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var data []byte
	if strings.HasPrefix(raw, "@") {
		path := raw[1:]
		if strings.HasPrefix(path, "~") {
			home, _ := os.UserHomeDir()
			path = home + path[1:]
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read MCP config: %w", err)
		}
		data = b
	} else {
		data = []byte(raw)
	}
	var servers []ServerConfig
	if err := json.Unmarshal(data, &servers); err != nil {
		return nil, fmt.Errorf("parse MCP config: %w", err)
	}
	for i := range servers {
		if servers[i].Command == "" {
			return nil, fmt.Errorf("MCP server %d: missing command", i)
		}
	}
	return servers, nil
}

// Tool is a tool exposed by an MCP server.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      *int64 `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string { return fmt.Sprintf("mcp: %s (code %d)", e.Message, e.Code) }

// Client is a connection to one MCP server.
type Client struct {
	cfg ServerConfig

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	nextID atomic.Int64

	mu      sync.Mutex
	pending map[int64]chan rpcMessage
	tools   []Tool
	closed  bool
	done    chan struct{}
	writeMu sync.Mutex
}

// NewClient creates an unstarted client.
func NewClient(cfg ServerConfig) *Client {
	return &Client{cfg: cfg, pending: map[int64]chan rpcMessage{}, done: make(chan struct{})}
}

// Name returns the server's display name.
func (c *Client) Name() string { return c.cfg.DisplayName() }

// Start launches the server process, performs the initialize handshake and
// caches the tool list.
func (c *Client) Start(ctx context.Context) error {
	cmd := exec.Command(c.cfg.Command, c.cfg.Args...)
	cmd.Env = os.Environ()
	for k, v := range c.cfg.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stderr = &logWriter{name: c.Name()}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start MCP server %s: %w", c.Name(), err)
	}
	c.cmd = cmd
	c.stdin = stdin
	go c.readLoop(stdout)

	initParams := map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "ovi", "version": "0.1.0"},
	}
	if _, err := c.call(ctx, "initialize", initParams); err != nil {
		_ = c.Close()
		return fmt.Errorf("initialize MCP server %s: %w", c.Name(), err)
	}
	if err := c.notify("notifications/initialized", map[string]any{}); err != nil {
		_ = c.Close()
		return err
	}
	tools, err := c.listTools(ctx)
	if err != nil {
		_ = c.Close()
		return fmt.Errorf("list tools for MCP server %s: %w", c.Name(), err)
	}
	c.mu.Lock()
	c.tools = tools
	c.mu.Unlock()
	return nil
}

// Tools returns the cached tool list.
func (c *Client) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Tool(nil), c.tools...)
}

func (c *Client) listTools(ctx context.Context) ([]Tool, error) {
	var all []Tool
	params := map[string]any{}
	for {
		raw, err := c.call(ctx, "tools/list", params)
		if err != nil {
			return nil, err
		}
		var res struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor"`
		}
		if err := json.Unmarshal(raw, &res); err != nil {
			return nil, err
		}
		all = append(all, res.Tools...)
		if res.NextCursor == "" {
			return all, nil
		}
		params = map[string]any{"cursor": res.NextCursor}
	}
}

// CallTool invokes a tool and returns its text content.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	if args == nil {
		args = map[string]any{}
	}
	raw, err := c.call(ctx, "tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return "", err
	}
	var res struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return "", err
	}
	var parts []string
	for _, item := range res.Content {
		if item.Type == "text" {
			parts = append(parts, item.Text)
		} else {
			parts = append(parts, fmt.Sprintf("[%s content omitted]", item.Type))
		}
	}
	text := strings.Join(parts, "\n")
	if res.IsError {
		return text, fmt.Errorf("mcp tool %s failed: %s", name, text)
	}
	return text, nil
}

// Close terminates the server process.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.mu.Unlock()
	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	if c.cmd != nil && c.cmd.Process != nil {
		select {
		case <-c.done:
		case <-time.After(3 * time.Second):
			slog.Debug("MCP server did not stop cleanly, killing", "name", c.Name())
			_ = c.cmd.Process.Kill()
			<-c.done
		}
	}
	return nil
}

func (c *Client) send(msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if c.stdin == nil {
		return errors.New("mcp: not started")
	}
	_, err = c.stdin.Write(append(data, '\n'))
	return err
}

func (c *Client) notify(method string, params any) error {
	return c.send(rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
}

func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	ch := make(chan rpcMessage, 1)
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, errors.New("mcp: client closed")
	}
	c.pending[id] = ch
	c.mu.Unlock()
	if err := c.send(rpcRequest{JSONRPC: "2.0", ID: &id, Method: method, Params: params}); err != nil {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, err
	}
	select {
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, ctx.Err()
	case <-c.done:
		return nil, fmt.Errorf("mcp: server %s exited", c.Name())
	case msg := <-ch:
		if msg.Error != nil {
			return nil, msg.Error
		}
		return msg.Result, nil
	}
}

func (c *Client) readLoop(stdout io.Reader) {
	defer func() {
		if c.cmd != nil {
			_ = c.cmd.Wait()
		}
		c.mu.Lock()
		for id, ch := range c.pending {
			close(ch)
			delete(c.pending, id)
		}
		c.mu.Unlock()
		close(c.done)
	}()
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64<<10), 32<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var msg rpcMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			slog.Debug("MCP: non-JSON line from server", "name", c.Name(), "line", line)
			continue
		}
		if msg.Method != "" {
			c.handleServerMessage(msg)
			continue
		}
		var id int64
		if err := json.Unmarshal(msg.ID, &id); err != nil {
			continue
		}
		c.mu.Lock()
		ch, ok := c.pending[id]
		delete(c.pending, id)
		c.mu.Unlock()
		if ok {
			ch <- msg
		}
	}
}

// handleServerMessage deals with server-initiated requests and notifications.
func (c *Client) handleServerMessage(msg rpcMessage) {
	if len(msg.ID) == 0 || string(msg.ID) == "null" {
		switch msg.Method {
		case "notifications/tools/list_changed":
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				if tools, err := c.listTools(ctx); err == nil {
					c.mu.Lock()
					c.tools = tools
					c.mu.Unlock()
				}
			}()
		case "notifications/message":
			var p struct {
				Level string `json:"level"`
				Data  any    `json:"data"`
			}
			_ = json.Unmarshal(msg.Params, &p)
			slog.Debug("MCP server log", "name", c.Name(), "level", p.Level, "data", p.Data)
		}
		return
	}
	// Requests we do not support: answer with method-not-found so the
	// server does not hang.
	switch msg.Method {
	case "ping":
		_ = c.send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": map[string]any{}})
	case "roots/list":
		_ = c.send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": map[string]any{"roots": []any{}}})
	default:
		_ = c.send(map[string]any{
			"jsonrpc": "2.0", "id": msg.ID,
			"error": map[string]any{"code": -32601, "message": "method not supported: " + msg.Method},
		})
	}
}

type logWriter struct{ name string }

func (w *logWriter) Write(p []byte) (int, error) {
	for _, line := range strings.Split(strings.TrimRight(string(p), "\n"), "\n") {
		if line != "" {
			slog.Debug("MCP stderr", "name", w.name, "line", line)
		}
	}
	return len(p), nil
}
