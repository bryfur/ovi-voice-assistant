// Package llm is a minimal client for OpenAI-compatible chat completion and
// transcription endpoints (OpenAI, ollama, vLLM, LM Studio, ...).
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

// DefaultBaseURL is used when no base URL is configured.
const DefaultBaseURL = "https://api.openai.com/v1"

// Client talks to an OpenAI-compatible API.
type Client struct {
	BaseURL string
	APIKey  string
	HTTP    *http.Client
}

// New creates a client; empty values fall back to OpenAI defaults.
func New(baseURL, apiKey string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if apiKey == "" {
		apiKey = "not-set"
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		HTTP:    &http.Client{Timeout: 5 * time.Minute},
	}
}

// Message is a chat message.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall is a model-requested function call.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the function name and JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolDef describes a callable tool for the model.
type ToolDef struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef is the function schema inside a ToolDef.
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ChatRequest is a chat completion request.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []ToolDef `json:"tools,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

// Choice is one completion choice.
type Choice struct {
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse is a non-streaming completion response.
type ChatResponse struct {
	Choices []Choice `json:"choices"`
}

// Text returns the first choice's content.
func (r *ChatResponse) Text() string {
	if r == nil || len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.Content
}

// APIError is a non-2xx response.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("llm: HTTP %d: %s", e.Status, truncate(e.Body, 300))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func (c *Client) post(ctx context.Context, path string, body []byte, contentType string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return nil, &APIError{Status: resp.StatusCode, Body: string(b)}
	}
	return resp, nil
}

// Chat performs a non-streaming chat completion.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Stream = false
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.post(ctx, "/chat/completions", body, "application/json")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("llm: decode response: %w", err)
	}
	return &out, nil
}

// StreamResult is the assembled result of a streamed completion.
type StreamResult struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
}

// AssistantMessage converts the result into a message for the history.
func (r *StreamResult) AssistantMessage() Message {
	return Message{Role: "assistant", Content: r.Content, ToolCalls: r.ToolCalls}
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content   *string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ChatStream performs a streaming chat completion, invoking onDelta for
// each content token. Tool calls are assembled and returned.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest, onDelta func(string)) (*StreamResult, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := c.post(ctx, "/chat/completions", body, "application/json")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	result := &StreamResult{}
	calls := map[int]*ToolCall{}
	var order []int
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64<<10), 8<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("llm: decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return nil, errors.New("llm: " + chunk.Error.Message)
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != nil && *ch.Delta.Content != "" {
				result.Content += *ch.Delta.Content
				if onDelta != nil {
					onDelta(*ch.Delta.Content)
				}
			}
			for _, tc := range ch.Delta.ToolCalls {
				call, ok := calls[tc.Index]
				if !ok {
					call = &ToolCall{Type: "function"}
					calls[tc.Index] = call
					order = append(order, tc.Index)
				}
				if tc.ID != "" {
					call.ID = tc.ID
				}
				if tc.Type != "" {
					call.Type = tc.Type
				}
				if tc.Function.Name != "" {
					call.Function.Name += tc.Function.Name
				}
				call.Function.Arguments += tc.Function.Arguments
			}
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				result.FinishReason = *ch.FinishReason
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("llm: read stream: %w", err)
	}
	for i, idx := range order {
		call := calls[idx]
		if call.ID == "" {
			call.ID = fmt.Sprintf("call_%d", i)
		}
		result.ToolCalls = append(result.ToolCalls, *call)
	}
	return result, nil
}

// Transcribe sends a WAV file to the /audio/transcriptions endpoint.
func (c *Client) Transcribe(ctx context.Context, wav []byte, model, language string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", err
	}
	if _, err := part.Write(wav); err != nil {
		return "", err
	}
	_ = w.WriteField("model", model)
	if language != "" {
		_ = w.WriteField("language", language)
	}
	_ = w.WriteField("response_format", "json")
	if err := w.Close(); err != nil {
		return "", err
	}
	resp, err := c.post(ctx, "/audio/transcriptions", buf.Bytes(), w.FormDataContentType())
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("llm: decode transcription: %w", err)
	}
	return strings.TrimSpace(out.Text), nil
}
