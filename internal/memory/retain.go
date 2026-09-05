package memory

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/llm"
)

// ChatCompleter is the subset of the LLM client used for fact extraction.
type ChatCompleter interface {
	Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
}

const extractSystem = `You extract structured facts from conversation text.

For each distinct fact, output a JSON object with these fields:
- "what": The core fact in 1-2 sentences. Resolve pronouns to names.
- "who": People involved (empty string if none).
- "where": Location (empty string if none).
- "when": Temporal info (empty string if none). Convert relative dates using today's date: %s.
- "why": Context or significance (empty string if none).
- "fact_type": One of "world", "experience", or "assistant".
  - "world": objective facts about the world
  - "experience": user's experiences, preferences, or personal info
  - "assistant": things the assistant said or did
- "confidence": 0.0 to 1.0, how confident this fact is.
- "entities": List of {"name": "...", "type": "person|organization|location|concept|other"}.

Only extract facts worth remembering long-term. Skip pleasantries, filler, and transient statements.

Output a JSON array of fact objects. If nothing is worth extracting, output [].
`

const extractUser = "Extract facts from this conversation:\n\n%s"

type rawEntity struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

type rawFact struct {
	What       string      `json:"what"`
	Who        string      `json:"who"`
	Where      string      `json:"where"`
	When       string      `json:"when"`
	Why        string      `json:"why"`
	FactType   string      `json:"fact_type"`
	Confidence *float64    `json:"confidence"`
	Entities   []rawEntity `json:"entities"`
}

// BuildFactText combines fact dimensions into a single text for embedding.
func BuildFactText(what, who, where, when, why string) string {
	parts := []string{what}
	for _, kv := range []struct{ k, v string }{{"who", who}, {"where", where}, {"when", when}, {"why", why}} {
		if v := strings.TrimSpace(kv.v); v != "" {
			parts = append(parts, kv.k+": "+v)
		}
	}
	return strings.Join(parts, ". ")
}

// stripCodeFences removes ```json fences around an LLM response.
func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if i := strings.Index(s, "\n"); i >= 0 {
		s = s[i+1:]
	} else {
		s = s[3:]
	}
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	return strings.TrimSpace(s)
}

// ParseFacts parses the LLM's JSON output (array or single object).
func ParseFacts(raw string) ([]rawFact, error) {
	cleaned := stripCodeFences(raw)
	var facts []rawFact
	if err := json.Unmarshal([]byte(cleaned), &facts); err == nil {
		return facts, nil
	}
	var single rawFact
	if err := json.Unmarshal([]byte(cleaned), &single); err != nil {
		return nil, err
	}
	return []rawFact{single}, nil
}

func newID12() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// Retain extracts facts from content and stores them in memory.
func Retain(ctx context.Context, bankID, content string, chat ChatCompleter, model string, embedder Embedder, store *MemoryStore, extraContext string) (RetainResult, error) {
	today := time.Now().UTC().Format("2006-01-02")
	userPrompt := fmt.Sprintf(extractUser, content)
	if extraContext != "" {
		userPrompt += "\n\nContext: " + extraContext
	}
	temp := 0.0
	resp, err := chat.Chat(ctx, llm.ChatRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: "system", Content: fmt.Sprintf(extractSystem, today)},
			{Role: "user", Content: userPrompt},
		},
		Temperature: &temp,
	})
	if err != nil {
		slog.Error("LLM fact extraction failed", "err", err)
		return RetainResult{Success: false}, nil
	}
	rawText := resp.Text()
	if rawText == "" {
		rawText = "[]"
	}
	rawFacts, err := ParseFacts(rawText)
	if err != nil {
		slog.Warn("Failed to parse LLM response as JSON", "response", truncate(rawText, 200))
		return RetainResult{Success: false}, nil
	}
	if len(rawFacts) == 0 {
		return RetainResult{Success: true, FactsCount: 0}, nil
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)
	facts := make([]Fact, 0, len(rawFacts))
	type entRef struct {
		factID string
		ent    rawEntity
	}
	var allEntities []entRef
	for _, r := range rawFacts {
		id := newID12()
		conf := 1.0
		if r.Confidence != nil {
			conf = *r.Confidence
		}
		facts = append(facts, Fact{
			ID:         id,
			BankID:     bankID,
			Text:       BuildFactText(r.What, r.Who, r.Where, r.When, r.Why),
			What:       r.What,
			Who:        r.Who,
			Where:      r.Where,
			When:       r.When,
			Why:        r.Why,
			FactType:   ParseFactType(r.FactType),
			Confidence: conf,
			CreatedAt:  now,
			OccurredAt: r.When,
		})
		for _, e := range r.Entities {
			allEntities = append(allEntities, entRef{id, e})
		}
	}

	// Embed all fact texts in one batch
	texts := make([]string, len(facts))
	for i, f := range facts {
		texts[i] = f.Text
	}
	if embs, err := embedder.Embed(ctx, texts); err != nil {
		slog.Error("Embedding failed, storing facts without embeddings", "err", err)
	} else if len(embs) == len(facts) {
		for i := range facts {
			facts[i].Embedding = embs[i]
		}
	}
	if err := store.SaveFacts(facts); err != nil {
		return RetainResult{Success: false}, err
	}
	factIDs := make([]string, len(facts))
	for i, f := range facts {
		factIDs[i] = f.ID
	}

	// Resolve and store entities
	type keyed struct {
		name    string
		etype   EntityType
		factIDs []string
	}
	merged := map[string]*keyed{}
	var order []string
	for _, ref := range allEntities {
		name := strings.TrimSpace(ref.ent.Name)
		if name == "" {
			continue
		}
		etype := ParseEntityType(strings.ToLower(ref.ent.Type))
		key := strings.ToLower(name) + "|" + string(etype)
		if k, ok := merged[key]; ok {
			k.factIDs = append(k.factIDs, ref.factID)
		} else {
			merged[key] = &keyed{name, etype, []string{ref.factID}}
			order = append(order, key)
		}
	}
	var toSave []Entity
	var entityIDs []string
	var toEmbed []int
	for _, key := range order {
		k := merged[key]
		existing, err := store.GetEntityByText(bankID, strings.ToLower(k.name), k.etype)
		if err != nil {
			return RetainResult{Success: false}, err
		}
		if existing != nil {
			set := map[string]bool{}
			for _, id := range existing.FactIDs {
				set[id] = true
			}
			for _, id := range k.factIDs {
				set[id] = true
			}
			ids := make([]string, 0, len(set))
			for id := range set {
				ids = append(ids, id)
			}
			existing.FactIDs = ids
			toSave = append(toSave, *existing)
			entityIDs = append(entityIDs, existing.ID)
		} else {
			e := Entity{
				ID:         newID12(),
				BankID:     bankID,
				Text:       k.name,
				EntityType: k.etype,
				FactIDs:    k.factIDs,
				CreatedAt:  now,
			}
			toSave = append(toSave, e)
			entityIDs = append(entityIDs, e.ID)
		}
	}
	for i := range toSave {
		if len(toSave[i].Embedding) == 0 {
			toEmbed = append(toEmbed, i)
		}
	}
	if len(toEmbed) > 0 {
		texts := make([]string, len(toEmbed))
		for i, idx := range toEmbed {
			texts[i] = toSave[idx].Text
		}
		if embs, err := embedder.Embed(ctx, texts); err != nil {
			slog.Error("Entity embedding failed", "err", err)
		} else if len(embs) == len(toEmbed) {
			for i, idx := range toEmbed {
				toSave[idx].Embedding = embs[i]
			}
		}
	}
	if len(toSave) > 0 {
		if err := store.SaveEntities(toSave); err != nil {
			return RetainResult{Success: false}, err
		}
	}
	slog.Info("Retained facts", "facts", len(facts), "entities", len(entityIDs), "bank", bankID)
	return RetainResult{Success: true, FactsCount: len(facts), FactIDs: factIDs, EntityIDs: entityIDs}, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
