package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// TestMain lets the test binary double as a fake MCP server when
// GO_FAKE_MCP_SERVER is set, so the client is exercised over real pipes.
func TestMain(m *testing.M) {
	if os.Getenv("GO_FAKE_MCP_SERVER") == "1" {
		fakeServer()
		return
	}
	os.Exit(m.Run())
}

func fakeServer() {
	sc := bufio.NewScanner(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	reply := func(id json.RawMessage, result any) {
		enc.Encode(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
	}
	for sc.Scan() {
		var m message
		if json.Unmarshal(sc.Bytes(), &m) != nil {
			continue
		}
		var p struct {
			Cursor string         `json:"cursor"`
			Name   string         `json:"name"`
			Args   map[string]any `json:"arguments"`
		}
		json.Unmarshal(m.Params, &p)
		switch m.Method {
		case "initialize":
			reply(m.ID, map[string]any{"protocolVersion": ProtocolVersion, "capabilities": map[string]any{},
				"serverInfo": map[string]any{"name": "fake"}})
		case "notifications/initialized":
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": 999, "method": "ping"}) // the client must answer
		case "tools/list":
			if p.Cursor == "" {
				reply(m.ID, map[string]any{"nextCursor": "page2", "tools": []map[string]any{{"name": "echo", "description": "Echo input",
					"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}}}}})
			} else {
				reply(m.ID, map[string]any{"tools": []map[string]any{{"name": "fail", "description": "Always fails"}}})
			}
		case "tools/call":
			if p.Name == "fail" {
				reply(m.ID, map[string]any{"content": []map[string]any{{"type": "text", "text": "nope"}}, "isError": true})
			} else {
				reply(m.ID, map[string]any{"content": []map[string]any{
					{"type": "text", "text": "echo: " + p.Args["text"].(string)}, {"type": "image", "data": "..."}}})
			}
		default:
			if len(m.ID) > 0 {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "error": map[string]any{"code": -32601, "message": "unknown method"}})
			}
		}
	}
}

func fakeClient(t *testing.T) *Client {
	t.Helper()
	c := NewClient(Server{Name: "fake", Command: os.Args[0], Env: map[string]string{"GO_FAKE_MCP_SERVER": "1"}})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestServerString(t *testing.T) {
	if (Server{Name: "n", Command: "c"}).String() != "n" || (Server{Command: "npx", Args: []string{"-y", "srv"}}).String() != "srv" ||
		(Server{Command: "c"}).String() != "c" {
		t.Fatal("naming wrong")
	}
}

func TestStartListsToolsAcrossPages(t *testing.T) {
	c := fakeClient(t)

	tools := c.Tools()

	if len(tools) != 2 || tools[0].Name != "echo" || tools[1].Name != "fail" || tools[0].InputSchema["type"] != "object" {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestCallReturnsTextAndReportsToolErrors(t *testing.T) {
	c := fakeClient(t)

	out, err := c.Call(context.Background(), "echo", map[string]any{"text": "hi"})
	failed, ferr := c.Call(context.Background(), "fail", nil)

	if err != nil || !strings.HasPrefix(out, "echo: hi") || !strings.Contains(out, "image content omitted") {
		t.Fatalf("got %q, %v", out, err)
	}
	if ferr == nil || failed != "nope" {
		t.Fatalf("got %q, %v", failed, ferr)
	}
}

func TestUnknownMethodIsAnError(t *testing.T) {
	c := fakeClient(t)

	_, err := c.request(context.Background(), "bogus", nil)

	if err == nil || !strings.Contains(err.Error(), "unknown method") {
		t.Fatalf("got %v", err)
	}
}

func TestCloseIsIdempotentAndCallsFailAfter(t *testing.T) {
	c := fakeClient(t)

	c.Close()
	c.Close()
	_, err := c.Call(context.Background(), "echo", nil)

	if err == nil {
		t.Fatal("expected error after close")
	}
}

func TestStartMissingBinary(t *testing.T) {
	c := NewClient(Server{Command: "/nonexistent/mcp-server"})

	if err := c.Start(context.Background()); err == nil {
		t.Fatal("expected error")
	}
}
