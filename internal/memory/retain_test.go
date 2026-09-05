package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/llm"
)

type fakeChat struct {
	content string
	err     error
	calls   int
}

func (f *fakeChat) Chat(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return &llm.ChatResponse{Choices: []llm.Choice{{Message: llm.Message{Content: f.content}}}}, nil
}

const extractJSON = `[
  {
    "what": "Alice is allergic to shellfish",
    "who": "Alice",
    "where": "",
    "when": "",
    "why": "dietary restriction",
    "fact_type": "experience",
    "confidence": 0.95,
    "entities": [{"name": "Alice", "type": "person"}]
  }
]`

func TestBuildFactText(t *testing.T) {
	got := BuildFactText("Alice likes pizza", "Alice", "NYC", "", "")

	if got != "Alice likes pizza. who: Alice. where: NYC" {
		t.Fatalf("got %q", got)
	}
	if BuildFactText("Simple fact", "", "", "", "") != "Simple fact" {
		t.Fatal("minimal text wrong")
	}
}

func TestParseFactsStripsFencesAndAcceptsObject(t *testing.T) {
	facts, err := ParseFacts("```json\n" + extractJSON + "\n```")
	if err != nil || len(facts) != 1 || facts[0].Who != "Alice" {
		t.Fatalf("fenced: %+v, %v", facts, err)
	}

	single, err := ParseFacts(`{"what":"x"}`)

	if err != nil || len(single) != 1 {
		t.Fatalf("single: %+v, %v", single, err)
	}
}

func TestRetainExtractsFactsAndEntities(t *testing.T) {
	s := openStore(t)
	emb := &fakeEmbedder{vec: []float32{0.1, 0.2, 0.3}}

	res, err := Retain(context.Background(), "test", "Alice told me she's allergic to shellfish", &fakeChat{content: extractJSON}, "m", emb, s, "")

	if err != nil || !res.Success || res.FactsCount != 1 || len(res.FactIDs) != 1 || len(res.EntityIDs) != 1 {
		t.Fatalf("got %+v, %v", res, err)
	}
	facts, _ := s.GetFacts("test", nil)
	if len(facts) != 1 || facts[0].FactType != FactExperience || facts[0].Confidence != 0.95 || len(facts[0].Embedding) != 3 {
		t.Fatalf("stored fact = %+v", facts[0])
	}
	entities, _ := s.GetEntities("test")
	if len(entities) != 1 || entities[0].Text != "Alice" || entities[0].EntityType != EntityPerson {
		t.Fatalf("entities = %+v", entities)
	}
}

func TestRetainEmptyExtraction(t *testing.T) {
	s := openStore(t)

	res, err := Retain(context.Background(), "test", "Hello", &fakeChat{content: "[]"}, "m", &fakeEmbedder{}, s, "")

	if err != nil || !res.Success || res.FactsCount != 0 {
		t.Fatalf("got %+v, %v", res, err)
	}
}

func TestRetainHandlesLLMFailureAndInvalidJSON(t *testing.T) {
	s := openStore(t)

	failed, err1 := Retain(context.Background(), "test", "x", &fakeChat{err: errors.New("api")}, "m", &fakeEmbedder{}, s, "")
	invalid, err2 := Retain(context.Background(), "test", "x", &fakeChat{content: "not json"}, "m", &fakeEmbedder{}, s, "")

	if err1 != nil || err2 != nil || failed.Success || invalid.Success {
		t.Fatalf("got %+v/%v %+v/%v", failed, err1, invalid, err2)
	}
}

func TestRetainEntityDedup(t *testing.T) {
	s := openStore(t)
	emb := &fakeEmbedder{vec: []float32{0.1, 0.2, 0.3}}

	Retain(context.Background(), "test", "first", &fakeChat{content: extractJSON}, "m", emb, s, "")
	Retain(context.Background(), "test", "second", &fakeChat{content: extractJSON}, "m", emb, s, "")

	entities, _ := s.GetEntities("test")
	if len(entities) != 1 || len(entities[0].FactIDs) != 2 {
		t.Fatalf("entities = %+v", entities)
	}
}
