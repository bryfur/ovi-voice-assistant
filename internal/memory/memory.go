package memory

import (
	"context"
	"errors"
	"log/slog"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/llm"
)

// Memory is the high-level memory interface wrapping store + embedder + LLM.
//
//	mem := memory.New(settings)
//	mem.Load()
//	mem.Retain(ctx, "Alice is allergic to shellfish", "")
//	facts, _ := mem.Recall(ctx, "What are Alice's allergies?", memory.BudgetMid, 1024)
//	mem.Close()
type Memory struct {
	settings *config.Settings
	bankID   string

	store    *MemoryStore
	embedder Embedder
	chat     ChatCompleter

	// NewEmbedder constructs the embedder; tests may override it.
	NewEmbedder func(model string) (Embedder, error)
}

// New creates an unloaded memory.
func New(settings *config.Settings) *Memory {
	return &Memory{
		settings: settings,
		bankID:   settings.Memory.BankID,
		NewEmbedder: func(model string) (Embedder, error) {
			e := NewONNXEmbedder(model)
			if err := e.Load(); err != nil {
				return nil, err
			}
			return e, nil
		},
	}
}

// BankID returns the memory partition name.
func (m *Memory) BankID() string { return m.bankID }

// Load opens the SQLite store and loads the local embedding model.
func (m *Memory) Load() error {
	raw := m.settings.Memory.DBPath
	dbPath := raw
	if raw != ":memory:" {
		dbPath = config.ExpandUser(raw)
	}
	store := NewMemoryStore(dbPath)
	if err := store.Open(); err != nil {
		return err
	}
	emb, err := m.NewEmbedder(m.settings.Memory.EmbeddingModel)
	if err != nil {
		store.Close()
		return err
	}
	m.store = store
	m.embedder = emb
	m.chat = llm.New(m.settings.LLM.BaseURL, m.settings.LLM.APIKey)
	slog.Info("Memory loaded", "db", dbPath, "embedding", m.settings.Memory.EmbeddingModel)
	return nil
}

// Close closes the store. It is idempotent.
func (m *Memory) Close() {
	if m.store != nil {
		_ = m.store.Close()
		m.store = nil
	}
	if c, ok := m.embedder.(interface{ Close() }); ok {
		c.Close()
	}
	slog.Info("Memory closed")
}

var errNotLoaded = errors.New("call Load() first")

// Retain extracts and stores facts from text.
func (m *Memory) Retain(ctx context.Context, content, extraContext string) (RetainResult, error) {
	if m.store == nil || m.embedder == nil || m.chat == nil {
		return RetainResult{}, errNotLoaded
	}
	return Retain(ctx, m.bankID, content, m.chat, m.settings.LLM.Model, m.embedder, m.store, extraContext)
}

// Recall searches memory for relevant facts.
func (m *Memory) Recall(ctx context.Context, query string, budget Budget, maxTokens int) (RecallResult, error) {
	if m.store == nil || m.embedder == nil {
		return RecallResult{}, errNotLoaded
	}
	if budget == "" {
		budget = BudgetMid
	}
	return Recall(ctx, m.bankID, query, m.embedder, m.store, budget, maxTokens)
}
