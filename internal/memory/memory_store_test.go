package memory

import "testing"

func openStore(t *testing.T) *MemoryStore {
	t.Helper()
	s := NewMemoryStore(":memory:")
	if err := s.Open(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func fact(id, text string, emb []float32) Fact {
	return Fact{ID: id, BankID: "test", Text: text, What: text, Embedding: emb, CreatedAt: "2025-01-01T00:00:00Z"}
}

func TestSaveAndGetFacts(t *testing.T) {
	s := openStore(t)

	err := s.SaveFacts([]Fact{fact("1", "Alice likes pizza", []float32{1, 0}), fact("2", "Bob likes tacos", nil)})
	facts, gerr := s.GetFacts("test", nil)

	if err != nil || gerr != nil || len(facts) != 2 {
		t.Fatalf("got %v %v %v", facts, err, gerr)
	}
	byID, _ := s.GetFacts("test", []string{"2"})
	if len(byID) != 1 || byID[0].Text != "Bob likes tacos" || byID[0].FactType != FactWorld {
		t.Fatalf("byID = %+v", byID)
	}
}

func TestSaveFactsUpserts(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{fact("1", "v1", nil)})

	s.SaveFacts([]Fact{fact("1", "v2", nil)})

	facts, _ := s.GetFacts("test", nil)
	if len(facts) != 1 || facts[0].Text != "v2" {
		t.Fatalf("got %+v", facts)
	}
}

func TestSearchFactsByEmbedding(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{
		fact("a", "a", []float32{1, 0, 0}),
		fact("b", "b", []float32{0, 1, 0}),
		fact("c", "c", nil),
	})

	scored, err := s.SearchFactsByEmbedding("test", []float32{0.9, 0.1, 0}, 10)

	if err != nil || len(scored) != 2 || scored[0].Fact.ID != "a" || scored[0].Score <= scored[1].Score {
		t.Fatalf("got %+v, %v", scored, err)
	}
}

func TestSearchFactsByEmbeddingEmptyQuery(t *testing.T) {
	s := openStore(t)

	scored, err := s.SearchFactsByEmbedding("test", nil, 10)

	if err != nil || len(scored) != 0 {
		t.Fatalf("got %v, %v", scored, err)
	}
}

func TestSearchFactsByText(t *testing.T) {
	s := openStore(t)
	s.SaveFacts([]Fact{fact("1", "Alice likes pizza", nil), fact("2", "Bob likes tacos", nil)})

	facts, err := s.SearchFactsByText("test", "PIZZA", 10)

	if err != nil || len(facts) != 1 || facts[0].ID != "1" {
		t.Fatalf("got %+v, %v", facts, err)
	}
}

func TestEntitiesRoundTripAndLookup(t *testing.T) {
	s := openStore(t)
	e := Entity{ID: "e1", BankID: "test", Text: "Alice", EntityType: EntityPerson, FactIDs: []string{"1"}, CreatedAt: "2025-01-01T00:00:00Z"}
	s.SaveFacts([]Fact{fact("1", "Alice works at Google", nil)})

	if err := s.SaveEntities([]Entity{e}); err != nil {
		t.Fatal(err)
	}
	got, err := s.GetEntityByText("test", "alice", EntityPerson)

	if err != nil || got == nil || got.ID != "e1" || len(got.FactIDs) != 1 {
		t.Fatalf("got %+v, %v", got, err)
	}
	if missing, _ := s.GetEntityByText("test", "nobody", ""); missing != nil {
		t.Fatal("expected nil for unknown entity")
	}
	facts, _ := s.GetFactsForEntity("test", "Alice")
	if len(facts) != 1 || facts[0].ID != "1" {
		t.Fatalf("facts for entity = %+v", facts)
	}
}

func TestCosineSimilarity(t *testing.T) {
	if CosineSimilarity([]float32{1, 0}, []float32{1, 0}) < 0.999 || CosineSimilarity([]float32{1, 0}, []float32{0, 1}) != 0 {
		t.Fatal("cosine wrong")
	}
	if CosineSimilarity([]float32{1}, []float32{1, 2}) != 0 {
		t.Fatal("mismatched lengths should be 0")
	}
}

func TestStoreNotOpened(t *testing.T) {
	s := NewMemoryStore(":memory:")

	_, err := s.GetFacts("test", nil)

	if err == nil {
		t.Fatal("expected error")
	}
}
