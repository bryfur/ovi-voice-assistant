package memory

import (
	"context"
	"testing"
)

type fakeEmbedder struct {
	vec []float32
	err error
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = f.vec
	}
	return out, nil
}

func (f *fakeEmbedder) EmbedOne(ctx context.Context, text string) ([]float32, error) {
	return EmbedOneWith(ctx, f, text)
}

func TestRRFMergeBasic(t *testing.T) {
	scores := rrfMerge(
		[]rankedID{{"a", 0.9}, {"b", 0.8}},
		[]rankedID{{"b", 1.0}, {"c", 0.5}},
	)

	if scores["b"] <= scores["a"] || scores["b"] <= scores["c"] {
		t.Fatalf("got %v", scores)
	}
}

func TestRRFMergeEmpty(t *testing.T) {
	if len(rrfMerge(nil, nil)) != 0 {
		t.Fatal("expected empty")
	}
}

func TestRecallReturnsRankedFacts(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{
		fact("1", "Alice likes pizza", []float32{1, 0, 0}),
		fact("2", "Bob likes tacos", []float32{0, 1, 0}),
	})

	res, err := Recall(context.Background(), "test", "alice pizza", &fakeEmbedder{vec: []float32{1, 0, 0}}, s, BudgetLow, 1024)

	if err != nil || len(res.Results) == 0 || res.Results[0].ID != "1" {
		t.Fatalf("got %+v, %v", res, err)
	}
	if res.Results[0].Relevance <= 0 {
		t.Fatal("relevance should be set")
	}
}

func TestRecallEmptyStore(t *testing.T) {
	s := openStore(t)

	res, err := Recall(context.Background(), "test", "anything", &fakeEmbedder{vec: []float32{1}}, s, BudgetLow, 1024)

	if err != nil || len(res.Results) != 0 || res.TotalCandidates != 0 {
		t.Fatalf("got %+v, %v", res, err)
	}
}

func TestRecallEntityGraph(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{
		fact("1", "Alice works at Google", []float32{0.5, 0.5, 0}),
		fact("2", "Bob works at Meta", []float32{0, 0.5, 0.5}),
	})
	s.SaveEntities([]Entity{{ID: "e1", BankID: "test", Text: "Alice", EntityType: EntityPerson, FactIDs: []string{"1"}, CreatedAt: "2025-01-01T00:00:00Z"}})

	res, err := Recall(context.Background(), "test", "Tell me about Alice", &fakeEmbedder{vec: []float32{1, 0, 0}}, s, BudgetLow, 1024)

	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, f := range res.Results {
		if f.ID == "1" {
			found = true
		}
	}
	if !found || len(res.Entities) != 1 || res.Entities[0].Text != "Alice" {
		t.Fatalf("got %+v", res)
	}
}

func TestRecallTokenBudget(t *testing.T) {
	s := openStore(t)
	var facts []Fact
	for i := 0; i < 20; i++ {
		text := ""
		for j := 0; j < 50; j++ {
			text += "fact number x "
		}
		facts = append(facts, fact(string(rune('a'+i)), text, []float32{1, 0, 0}))
	}
	s.SaveFacts(facts)

	res, err := Recall(context.Background(), "test", "fact", &fakeEmbedder{vec: []float32{1, 0, 0}}, s, BudgetHigh, 100)

	if err != nil || len(res.Results) >= 20 || res.TotalCandidates < len(res.Results) {
		t.Fatalf("got %d results / %d candidates, %v", len(res.Results), res.TotalCandidates, err)
	}
}

func TestRecallSurvivesEmbeddingFailure(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{fact("1", "Alice likes pizza", []float32{1, 0, 0})})

	res, err := Recall(context.Background(), "test", "pizza", &fakeEmbedder{err: context.DeadlineExceeded}, s, BudgetLow, 1024)

	if err != nil || len(res.Results) != 1 {
		t.Fatalf("got %+v, %v", res, err)
	}
}
