// Package mcp is a minimal Model Context Protocol client: newline-delimited
// JSON-RPC 2.0 over the stdio of a subprocess, with initialize, tools/list
// and tools/call.
package mcp

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ProtocolVersion is the MCP revision we advertise.
const ProtocolVersion = "2025-06-18"

// Server is how to launch an MCP server.
type Server struct {
	Name    string            `json:"name,omitempty"`
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// String is the configured name, else the last argument, else the command.
func (s Server) String() string {
	last := ""
	if len(s.Args) > 0 {
		last = s.Args[len(s.Args)-1]
	}
	return cmp.Or(s.Name, last, s.Command)
}

// Tool is a tool a server exposes.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type message struct {
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

func raw(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// Client talks to one server process.
type Client struct {
	Server

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	writes sync.Mutex
	nextID atomic.Int64

	mu      sync.Mutex
	pending map[int64]chan message
	tools   []Tool
	closed  bool
	exited  chan struct{}
}

// NewClient prepares a client; Start launches the server.
func NewClient(s Server) *Client {
	return &Client{Server: s, pending: map[int64]chan message{}, exited: make(chan struct{})}
}

// Start launches the server, performs the initialize handshake and
// fetches its tools.
func (c *Client) Start(ctx context.Context) error {
	cmd := exec.Command(c.Command, c.Args...)
	cmd.Env = os.Environ()
	for k, v := range c.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stderr = stderrLog{c.String()}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start MCP server %s: %w", c, err)
	}
	c.cmd, c.stdin = cmd, stdin
	go c.read(stdout)

	_, err = c.request(ctx, "initialize", map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "ovi", "version": "0.1.0"},
	})
	if err == nil {
		err = c.send(message{JSONRPC: "2.0", Method: "notifications/initialized", Params: raw(map[string]any{})})
	}
	if err == nil {
		err = c.refreshTools(ctx)
	}
	if err != nil {
		_ = c.Close()
		return fmt.Errorf("MCP server %s: %w", c, err)
	}
	return nil
}

// Tools lists the server's tools.
func (c *Client) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Tool(nil), c.tools...)
}

func (c *Client) refreshTools(ctx context.Context) error {
	var all []Tool
	params := map[string]any{}
	for {
		res, err := c.request(ctx, "tools/list", params)
		if err != nil {
			return err
		}
		var page struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor"`
		}
		if err := json.Unmarshal(res, &page); err != nil {
			return err
		}
		all = append(all, page.Tools...)
		if page.NextCursor == "" {
			break
		}
		params = map[string]any{"cursor": page.NextCursor}
	}
	c.mu.Lock()
	c.tools = all
	c.mu.Unlock()
	return nil
}

// Call invokes a tool and returns its text content; a tool-reported
// failure comes back as both text and error.
func (c *Client) Call(ctx context.Context, name string, args map[string]any) (string, error) {
	if args == nil {
		args = map[string]any{}
	}
	res, err := c.request(ctx, "tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return "", err
	}
	var out struct {
		Content []struct{ Type, Text string } `json:"content"`
		IsError bool                          `json:"isError"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return "", err
	}
	parts := make([]string, len(out.Content))
	for i, item := range out.Content {
		parts[i] = item.Text
		if item.Type != "text" {
			parts[i] = fmt.Sprintf("[%s content omitted]", item.Type)
		}
	}
	text := strings.Join(parts, "\n")
	if out.IsError {
		return text, fmt.Errorf("mcp tool %s failed: %s", name, text)
	}
	return text, nil
}

// Close stops the server, killing it if it ignores a closed stdin.
func (c *Client) Close() error {
	c.mu.Lock()
	closed := c.closed
	c.closed = true
	c.mu.Unlock()
	if closed || c.cmd == nil {
		return nil
	}
	_ = c.stdin.Close()
	select {
	case <-c.exited:
	case <-time.After(3 * time.Second):
		slog.Debug("MCP server did not stop, killing", "name", c.String())
		_ = c.cmd.Process.Kill()
		<-c.exited
	}
	return nil
}

func (c *Client) send(m message) error {
	c.writes.Lock()
	defer c.writes.Unlock()
	if c.stdin == nil {
		return errors.New("mcp: not started")
	}
	_, err := c.stdin.Write(append(raw(m), '\n'))
	return err
}

// request sends a call and waits for its reply.
func (c *Client) request(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	reply := make(chan message, 1)
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, errors.New("mcp: client closed")
	}
	c.pending[id] = reply
	c.mu.Unlock()
	forget := func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}
	m := message{JSONRPC: "2.0", ID: json.RawMessage(strconv.FormatInt(id, 10)), Method: method, Params: raw(params)}
	if err := c.send(m); err != nil {
		forget()
		return nil, err
	}
	select {
	case <-ctx.Done():
		forget()
		return nil, ctx.Err()
	case <-c.exited:
		return nil, fmt.Errorf("mcp: server %s exited", c)
	case m := <-reply:
		if m.Error != nil {
			return nil, m.Error
		}
		return m.Result, nil
	}
}

// read dispatches server output until the process exits.
func (c *Client) read(stdout io.Reader) {
	defer func() {
		_ = c.cmd.Wait()
		c.mu.Lock()
		for id, ch := range c.pending {
			close(ch)
			delete(c.pending, id)
		}
		c.mu.Unlock()
		close(c.exited)
	}()
	sc := bufio.NewScanner(stdout)
	sc.Buffer(nil, 32<<20)
	for sc.Scan() {
		var m message
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			slog.Debug("MCP: non-JSON line from server", "name", c.String(), "line", sc.Text())
			continue
		}
		if m.Method != "" {
			c.serve(m)
			continue
		}
		id, err := strconv.ParseInt(string(m.ID), 10, 64)
		if err != nil {
			continue
		}
		c.mu.Lock()
		reply, ok := c.pending[id]
		delete(c.pending, id)
		c.mu.Unlock()
		if ok {
			reply <- m
		}
	}
}

// serve answers server-initiated requests and notifications.
func (c *Client) serve(m message) {
	if len(m.ID) == 0 || string(m.ID) == "null" { // notification
		switch m.Method {
		case "notifications/tools/list_changed":
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				_ = c.refreshTools(ctx)
			}()
		case "notifications/message":
			slog.Debug("MCP server log", "name", c.String(), "params", string(m.Params))
		}
		return
	}
	reply := message{JSONRPC: "2.0", ID: m.ID}
	switch m.Method {
	case "ping":
		reply.Result = raw(map[string]any{})
	case "roots/list":
		reply.Result = raw(map[string]any{"roots": []any{}})
	default:
		reply.Error = &rpcError{Code: -32601, Message: "method not supported: " + m.Method}
	}
	_ = c.send(reply)
}

// stderrLog relays the server's stderr to the debug log.
type stderrLog struct{ name string }

func (l stderrLog) Write(p []byte) (int, error) {
	for line := range strings.SplitSeq(strings.TrimSpace(string(p)), "\n") {
		if line != "" {
			slog.Debug("MCP stderr", "name", l.name, "line", line)
		}
	}
	return len(p), nil
}
