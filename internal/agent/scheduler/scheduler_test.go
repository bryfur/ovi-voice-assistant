package scheduler

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type recorder struct {
	prompts, announced []string
	answer             string
}

func (r *recorder) ask(_ context.Context, prompt string) (string, error) {
	r.prompts = append(r.prompts, prompt)
	return r.answer, nil
}

func (r *recorder) announce(text string) { r.announced = append(r.announced, text) }

func newScheduler(t *testing.T) (*Scheduler, *recorder) {
	t.Helper()
	rec := &recorder{answer: "The weather is sunny."}
	return New(filepath.Join(t.TempDir(), "automations.json"), rec.ask, rec.announce), rec
}

func TestCreateListDeleteEnable(t *testing.T) {
	s, _ := newScheduler(t)

	a, err := s.Create("morning", "0 7 * * *", "weather?")
	_, bad := s.Create("bad", "0 7 * *", "x")

	if err != nil || a.ID == "" || !a.Enabled || bad == nil {
		t.Fatalf("create: %+v, %v, %v", a, err, bad)
	}
	if list := s.Automations(); len(list) != 1 || list[0].Name != "morning" {
		t.Fatalf("list = %+v", list)
	}
	if !s.Enable("morning", false) || s.Automations()[0].Enabled || s.Enable("missing", true) {
		t.Fatal("enable semantics wrong")
	}
	if !s.Delete("morning") || s.Delete("morning") || len(s.Automations()) != 0 {
		t.Fatal("delete semantics wrong")
	}
}

func TestPersistenceAndCorruptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.json")
	New(path, nil, nil).Create("a", "0 7 * * *", "prompt")

	reloaded := New(path, nil, nil).Automations()

	if len(reloaded) != 1 || reloaded[0].Prompt != "prompt" {
		t.Fatalf("reloaded = %+v", reloaded)
	}
	data, _ := os.ReadFile(path)
	for _, key := range []string{`"id"`, `"name"`, `"schedule"`, `"prompt"`, `"enabled"`, `"last_run"`} {
		if !strings.Contains(string(data), key) {
			t.Fatalf("json missing %s: %s", key, data)
		}
	}
	os.WriteFile(path, []byte("not json"), 0o644)
	if len(New(path, nil, nil).Automations()) != 0 {
		t.Fatal("corrupt file should yield an empty list")
	}
}

func TestFireAnnouncesNonEmptyAnswers(t *testing.T) {
	s, rec := newScheduler(t)

	s.Fire(context.Background(), Automation{Name: "a", Prompt: "weather?"})
	rec.answer = "   "
	s.Fire(context.Background(), Automation{Name: "b", Prompt: "x"})

	if len(rec.prompts) != 2 || rec.prompts[0] != "weather?" || len(rec.announced) != 1 {
		t.Fatalf("prompts=%v announced=%v", rec.prompts, rec.announced)
	}
}

func TestCheckFiresDueOnceSkippingDisabled(t *testing.T) {
	s, rec := newScheduler(t)
	s.Now = func() time.Time { return time.Date(2025, 1, 6, 7, 0, 10, 0, time.Local) }
	s.Create("match", "0 7 * * *", "a")
	s.Create("nomatch", "0 8 * * *", "b")
	s.Create("disabled", "0 7 * * *", "c")
	s.Enable("disabled", false)

	s.Check(context.Background())
	s.Check(context.Background()) // same minute: no double fire

	if len(rec.prompts) != 1 || rec.prompts[0] != "a" || s.Automations()[0].LastRun == "" {
		t.Fatalf("prompts = %v", rec.prompts)
	}
}

func TestStartStop(t *testing.T) {
	s, _ := newScheduler(t)

	s.Start()
	s.Start()
	s.Stop()
	s.Stop()
}
