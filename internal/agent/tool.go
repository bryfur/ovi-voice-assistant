package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// Tool is a function the model can call.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any // JSON schema; nil means none
	Run         func(ctx context.Context, env *Env, args Args) (string, error)
}

// Def is the tool as the API sees it.
func (t Tool) Def() openai.ChatCompletionToolUnionParam {
	params := t.Parameters
	if params == nil {
		params = schema(nil, map[string]any{})
	}
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        t.Name,
		Description: openai.String(t.Description),
		Parameters:  shared.FunctionParameters(params),
	})
}

// schema builds an object schema from its properties.
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

// Args are a tool call's decoded arguments.
type Args map[string]any

func parseArgs(raw string) (Args, error) {
	if strings.TrimSpace(raw) == "" {
		return Args{}, nil
	}
	var a Args
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		return nil, fmt.Errorf("invalid tool arguments: %w", err)
	}
	return a, nil
}

// String returns the argument as text, or def when missing.
func (a Args) String(key, def string) string {
	switch v := a[key].(type) {
	case nil:
		return def
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

// Float returns a numeric argument, or def when missing.
func (a Args) Float(key string, def float64) float64 {
	if v, ok := a[key].(float64); ok {
		return v
	}
	return def
}

// Int returns an integer argument, or def when missing.
func (a Args) Int(key string, def int) int { return int(a.Float(key, float64(def))) }

// Bool returns a boolean argument, or def when missing.
func (a Args) Bool(key string, def bool) bool {
	if v, ok := a[key].(bool); ok {
		return v
	}
	return def
}
