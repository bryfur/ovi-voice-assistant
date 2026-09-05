package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type recorder struct {
	mu        sync.Mutex
	prompts   []string
	announced []string
	response  string
}

func (r *recorder) run(_ context.Context, prompt string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prompts = append(r.prompts, prompt)
	return r.response, nil
}

func (r *recorder) announce(_ context.Context, text string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.announced = append(r.announced, text)
	return nil
}

func newScheduler(t *testing.T) (*Scheduler, *recorder) {
	t.Helper()
	rec := &recorder{response: "The weather is sunny."}
	s := NewScheduler(filepath.Join(t.TempDir(), "automations.json"), rec.run, rec.announce)
	s.Load()
	return s, rec
}

func TestCreateAndList(t *testing.T) {
	s, _ := newScheduler(t)

	auto, err := s.Create("morning", "0 7 * * *", "weather?")

	if err != nil || auto.ID == "" || !auto.Enabled {
		t.Fatalf("got %+v, %v", auto, err)
	}
	if list := s.Automations(); len(list) != 1 || list[0].Name != "morning" {
		t.Fatalf("list = %+v", list)
	}
}

func TestCreateInvalidCron(t *testing.T) {
	s, _ := newScheduler(t)

	_, err := s.Create("bad", "0 7 * *", "x")

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDeleteAndNotFound(t *testing.T) {
	s, _ := newScheduler(t)
	s.Create("a", "* * * * *", "x")

	if !s.Delete("a") || s.Delete("a") || len(s.Automations()) != 0 {
		t.Fatal("delete semantics wrong")
	}
}

func TestEnableDisable(t *testing.T) {
	s, _ := newScheduler(t)
	s.Create("a", "* * * * *", "x")

	ok := s.SetEnabled("a", false)

	if !ok || s.Automations()[0].Enabled || s.SetEnabled("missing", true) {
		t.Fatal("toggle semantics wrong")
	}
}

func TestPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.json")
	s := NewScheduler(path, nil, nil)
	s.Create("a", "0 7 * * *", "prompt")

	s2 := NewScheduler(path, nil, nil)
	s2.Load()

	if list := s2.Automations(); len(list) != 1 || list[0].Prompt != "prompt" {
		t.Fatalf("reloaded = %+v", list)
	}
	data, _ := os.ReadFile(path)
	for _, key := range []string{`"id"`, `"name"`, `"schedule"`, `"prompt"`, `"enabled"`, `"last_run"`} {
		if !strings.Contains(string(data), key) {
			t.Fatalf("json missing %s: %s", key, data)
		}
	}
}

func TestLoadCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.json")
	os.WriteFile(path, []byte("not json"), 0o644)
	s := NewScheduler(path, nil, nil)

	s.Load()

	if len(s.Automations()) != 0 {
		t.Fatal("corrupt file should yield empty list")
	}
}

func TestFireCallsPromptAndAnnounce(t *testing.T) {
	s, rec := newScheduler(t)

	s.Fire(context.Background(), Automation{Name: "a", Prompt: "weather?"})

	if len(rec.prompts) != 1 || rec.prompts[0] != "weather?" || len(rec.announced) != 1 {
		t.Fatalf("prompts=%v announced=%v", rec.prompts, rec.announced)
	}
}

func TestFireEmptyResponseDoesNotAnnounce(t *testing.T) {
	s, rec := newScheduler(t)
	rec.response = "   "

	s.Fire(context.Background(), Automation{Name: "a", Prompt: "x"})

	if len(rec.announced) != 0 {
		t.Fatal("empty response must not be announced")
	}
}

func TestCheckFiresMatchingSkipsDisabledAndPreventsDoubleFire(t *testing.T) {
	s, rec := newScheduler(t)
	s.Now = func() time.Time { return time.Date(2025, 1, 6, 7, 0, 10, 0, time.Local) }
	s.Create("match", "0 7 * * *", "a")
	s.Create("nomatch", "0 8 * * *", "b")
	s.Create("disabled", "0 7 * * *", "c")
	s.SetEnabled("disabled", false)

	s.Check(context.Background())
	s.Check(context.Background()) // same minute → no double fire
	s.fires.Wait()

	if len(rec.prompts) != 1 || rec.prompts[0] != "a" {
		t.Fatalf("prompts = %v", rec.prompts)
	}
	if lr := s.Automations()[0].LastRun; lr == "" {
		t.Fatal("last_run not recorded")
	}
}

func TestStartStop(t *testing.T) {
	s, _ := newScheduler(t)

	s.Start()
	s.Stop()
	s.Stop() // idempotent
}
