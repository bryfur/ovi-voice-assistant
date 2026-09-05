package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/bryfur/ovi-voice-assistant/internal/llm"
)

// Handler executes a tool with decoded JSON arguments.
type Handler func(ctx context.Context, actx *Context, args Args) (string, error)

// Tool is a callable function exposed to the model.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	// Enabled is false for tools that are defined but never offered to the
	// model (mirrors is_enabled=False in the original).
	Enabled bool
	Handler Handler
}

// Def converts the tool into an OpenAI function definition.
func (t Tool) Def() llm.ToolDef {
	params := t.Parameters
	if params == nil {
		params = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return llm.ToolDef{
		Type: "function",
		Function: llm.FunctionDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		},
	}
}

// Args holds decoded tool-call arguments.
type Args map[string]any

// ParseArgs decodes a JSON argument string; empty input yields empty args.
func ParseArgs(raw string) (Args, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Args{}, nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return nil, fmt.Errorf("invalid tool arguments: %w", err)
	}
	if m == nil {
		m = map[string]any{}
	}
	return m, nil
}

// String returns a string argument or the default.
func (a Args) String(key, def string) string {
	v, ok := a[key]
	if !ok || v == nil {
		return def
	}
	switch s := v.(type) {
	case string:
		return s
	case float64:
		if s == math.Trunc(s) {
			return fmt.Sprintf("%d", int64(s))
		}
		return fmt.Sprintf("%g", s)
	case bool:
		if s {
			return "true"
		}
		return "false"
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// Float returns a numeric argument or the default.
func (a Args) Float(key string, def float64) float64 {
	v, ok := a[key]
	if !ok || v == nil {
		return def
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case string:
		var f float64
		if _, err := fmt.Sscanf(strings.TrimSpace(n), "%g", &f); err == nil {
			return f
		}
	}
	return def
}

// Int returns an integer argument or the default.
func (a Args) Int(key string, def int) int {
	v, ok := a[key]
	if !ok || v == nil {
		return def
	}
	return int(a.Float(key, float64(def)))
}

// Bool returns a boolean argument or the default.
func (a Args) Bool(key string, def bool) bool {
	v, ok := a[key]
	if !ok || v == nil {
		return def
	}
	switch b := v.(type) {
	case bool:
		return b
	case string:
		switch strings.ToLower(b) {
		case "true", "1", "yes":
			return true
		case "false", "0", "no":
			return false
		}
	case float64:
		return b != 0
	}
	return def
}

// schema is a small helper to build JSON schemas.
func schema(required []string, props map[string]any) map[string]any {
	s := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		s["required"] = required
	}
	return s
}

func prop(typ, desc string) map[string]any {
	return map[string]any{"type": typ, "description": desc}
}
