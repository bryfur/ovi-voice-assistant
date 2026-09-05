package memory

import (
	"context"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func memSettings() *config.Settings {
	s := config.Default()
	s.Memory.DBPath = ":memory:"
	return s
}

func TestRetainRecallRaiseBeforeLoad(t *testing.T) {
	m := New(memSettings())

	_, err1 := m.Retain(context.Background(), "x", "")
	_, err2 := m.Recall(context.Background(), "x", BudgetMid, 10)

	if err1 == nil || err2 == nil {
		t.Fatal("expected errors before Load")
	}
}

func TestLoadOpensStoreAndCloseIsIdempotent(t *testing.T) {
	m := New(memSettings())
	loaded := false
	m.NewEmbedder = func(model string) (Embedder, error) {
		loaded = true
		return &fakeEmbedder{vec: []float32{1}}, nil
	}

	err := m.Load()

	if err != nil || !loaded || m.BankID() != "voice-assistant" {
		t.Fatalf("err=%v loaded=%v bank=%s", err, loaded, m.BankID())
	}
	m.Close()
	m.Close()
}

func TestRecallAfterLoad(t *testing.T) {
	m := New(memSettings())
	m.NewEmbedder = func(string) (Embedder, error) { return &fakeEmbedder{vec: []float32{1}}, nil }
	m.Load()
	defer m.Close()
	f := fact("1", "Alice likes pizza", []float32{1})
	f.BankID = m.BankID()
	m.store.SaveFacts([]Fact{f})

	res, err := m.Recall(context.Background(), "pizza", "", 100)

	if err != nil || len(res.Results) != 1 {
		t.Fatalf("got %+v, %v", res, err)
	}
}
