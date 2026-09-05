package agent

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
// GO_FAKE_MCP_SERVER is set, so the client can be exercised over real
// stdio pipes without external dependencies.
func TestMain(m *testing.M) {
	if os.Getenv("GO_FAKE_MCP_SERVER") == "1" {
		runFakeServer()
		return
	}
	os.Exit(m.Run())
}

func runFakeServer() {
	scanner := bufio.NewScanner(os.Stdin)
	enc := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var msg rpcMessage
		if json.Unmarshal(scanner.Bytes(), &msg) != nil {
			continue
		}
		switch msg.Method {
		case "initialize":
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": map[string]any{
				"protocolVersion": mcpProtocolVersion, "capabilities": map[string]any{},
				"serverInfo": map[string]any{"name": "fake"},
			}})
		case "notifications/initialized":
			// Also exercise a server→client request the client must answer.
			enc.Encode(map[string]any{"jsonrpc": "2.0", "id": 999, "method": "ping"})
		case "tools/list":
			var p struct {
				Cursor string `json:"cursor"`
			}
			json.Unmarshal(msg.Params, &p)
			if p.Cursor == "" {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": map[string]any{
					"tools": []map[string]any{{"name": "echo", "description": "Echo input",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}}}},
					"nextCursor": "page2",
				}})
			} else {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": map[string]any{
					"tools": []map[string]any{{"name": "fail", "description": "Always fails"}},
				}})
			}
		case "tools/call":
			var p struct {
				Name string         `json:"name"`
				Args map[string]any `json:"arguments"`
			}
			json.Unmarshal(msg.Params, &p)
			if p.Name == "fail" {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": map[string]any{
					"content": []map[string]any{{"type": "text", "text": "nope"}}, "isError": true}})
			} else {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID), "result": map[string]any{
					"content": []map[string]any{{"type": "text", "text": "echo: " + p.Args["text"].(string)},
						{"type": "image", "data": "..."}}}})
			}
		default:
			if len(msg.ID) > 0 {
				enc.Encode(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID),
					"error": map[string]any{"code": -32601, "message": "unknown method"}})
			}
		}
	}
}

func fakeClient(t *testing.T) *mcpClient {
	t.Helper()
	c := newMCPClient(MCPServer{Name: "fake", Command: os.Args[0], Env: map[string]string{"GO_FAKE_MCP_SERVER": "1"}})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestParseServersInlineAndFile(t *testing.T) {
	inline, err := parseMCPServers(`[{"command":"npx","args":["-y","x"]}]`)
	if err != nil || len(inline) != 1 || inline[0].DisplayName() != "x" {
		t.Fatalf("inline = %+v, %v", inline, err)
	}
	path := t.TempDir() + "/json"
	os.WriteFile(path, []byte(`[{"name":"weather","command":"w"}]`), 0o644)

	fromFile, err := parseMCPServers("@" + path)

	if err != nil || len(fromFile) != 1 || fromFile[0].DisplayName() != "weather" {
		t.Fatalf("file = %+v, %v", fromFile, err)
	}
}

func TestParseServersEmptyAndInvalid(t *testing.T) {
	empty, err := parseMCPServers("  ")
	if err != nil || empty != nil {
		t.Fatalf("empty = %v, %v", empty, err)
	}

	_, err = parseMCPServers(`[{"args":["x"]}]`)

	if err == nil || !strings.Contains(err.Error(), "missing command") {
		t.Fatalf("expected missing command error, got %v", err)
	}
}

func TestStartListsToolsAcrossPages(t *testing.T) {
	c := fakeClient(t)

	tools := c.Tools()

	if len(tools) != 2 || tools[0].Name != "echo" || tools[1].Name != "fail" {
		t.Fatalf("tools = %+v", tools)
	}
	if tools[0].InputSchema["type"] != "object" {
		t.Fatalf("schema = %v", tools[0].InputSchema)
	}
}

func TestCallToolReturnsText(t *testing.T) {
	c := fakeClient(t)

	out, err := c.CallTool(context.Background(), "echo", map[string]any{"text": "hi"})

	if err != nil || !strings.HasPrefix(out, "echo: hi") || !strings.Contains(out, "image content omitted") {
		t.Fatalf("got %q, %v", out, err)
	}
}

func TestCallToolIsError(t *testing.T) {
	c := fakeClient(t)

	out, err := c.CallTool(context.Background(), "fail", nil)

	if err == nil || out != "nope" {
		t.Fatalf("got %q, %v", out, err)
	}
}

func TestCallToolUnknownMethodError(t *testing.T) {
	c := fakeClient(t)

	_, err := c.call(context.Background(), "bogus", nil)

	if err == nil || !strings.Contains(err.Error(), "unknown method") {
		t.Fatalf("got %v", err)
	}
}

func TestCloseIsIdempotentAndCallsFailAfter(t *testing.T) {
	c := fakeClient(t)

	c.Close()
	c.Close()
	_, err := c.CallTool(context.Background(), "echo", nil)

	if err == nil {
		t.Fatal("expected error after close")
	}
}

func TestStartMissingBinary(t *testing.T) {
	c := newMCPClient(MCPServer{Command: "/nonexistent/mcp-server"})

	err := c.Start(context.Background())

	if err == nil {
		t.Fatal("expected error")
	}
}
