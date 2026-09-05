// Package scheduler runs cron automations: at the scheduled minute a
// prompt goes through the agent and the answer is announced on every
// device. Automations persist as JSON.
package scheduler

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// tick is how often due automations are looked for.
const tick = 30 * time.Second

// Automation is one scheduled prompt.
type Automation struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Schedule string `json:"schedule"` // five-field cron: minute hour day month weekday
	Prompt   string `json:"prompt"`
	Enabled  bool   `json:"enabled"`
	LastRun  string `json:"last_run"` // the minute it last fired, RFC 3339
}

// Ask runs a prompt through the agent.
type Ask func(ctx context.Context, prompt string) (string, error)

// Scheduler fires automations on time and keeps them on disk.
type Scheduler struct {
	path     string
	ask      Ask
	announce func(text string)
	// Now is the clock; tests override it.
	Now func() time.Time

	mu   sync.Mutex
	list []Automation
	stop context.CancelFunc
	done chan struct{}
}

// New loads the automations stored at path (none if it is missing or
// unreadable).
func New(path string, ask Ask, announce func(string)) *Scheduler {
	s := &Scheduler{path: path, ask: ask, announce: announce, Now: time.Now}
	data, err := os.ReadFile(path)
	if err == nil {
		err = json.Unmarshal(data, &s.list)
	}
	if err != nil && !os.IsNotExist(err) {
		slog.Error("Cannot load automations", "path", path, "err", err)
		s.list = nil
	}
	slog.Info("Automations loaded", "count", len(s.list), "path", path)
	return s
}

// Automations lists everything scheduled.
func (s *Scheduler) Automations() []Automation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.list)
}

// Create adds an enabled automation.
func (s *Scheduler) Create(name, schedule, prompt string) (Automation, error) {
	if len(strings.Fields(schedule)) != 5 {
		return Automation{}, fmt.Errorf("invalid cron expression %q: expected 5 fields (minute hour day-of-month month day-of-week)", schedule)
	}
	id := make([]byte, 4)
	_, _ = rand.Read(id)
	a := Automation{ID: hex.EncodeToString(id), Name: name, Schedule: schedule, Prompt: prompt, Enabled: true}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.list = append(s.list, a)
	s.save()
	slog.Info("Automation created", "name", name, "schedule", schedule)
	return a, nil
}

// Delete removes an automation by name.
func (s *Scheduler) Delete(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.list)
	s.list = slices.DeleteFunc(s.list, func(a Automation) bool { return a.Name == name })
	if len(s.list) == n {
		return false
	}
	s.save()
	slog.Info("Automation deleted", "name", name)
	return true
}

// Enable turns an automation on or off by name.
func (s *Scheduler) Enable(name string, on bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := slices.IndexFunc(s.list, func(a Automation) bool { return a.Name == name })
	if i < 0 {
		return false
	}
	s.list[i].Enabled = on
	s.save()
	slog.Info("Automation toggled", "name", name, "enabled", on)
	return true
}

// save writes the list; the caller holds mu.
func (s *Scheduler) save() {
	data, _ := json.MarshalIndent(append([]Automation{}, s.list...), "", "  ")
	err := os.MkdirAll(filepath.Dir(s.path), 0o755)
	if err == nil {
		err = os.WriteFile(s.path, data, 0o644)
	}
	if err != nil {
		slog.Error("Cannot save automations", "path", s.path, "err", err)
	}
}

// Start checks for due automations every tick until Stop.
func (s *Scheduler) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stop != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	s.stop, s.done = cancel, done
	go func() {
		defer close(done)
		t := time.NewTicker(tick)
		defer t.Stop()
		for {
			s.Check(ctx)
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
		}
	}()
	slog.Info("Scheduler started", "automations", len(s.list))
}

// Stop ends the checks and waits for a running automation to finish.
func (s *Scheduler) Stop() {
	s.mu.Lock()
	stop, done := s.stop, s.done
	s.stop, s.done = nil, nil
	s.mu.Unlock()
	if stop != nil {
		stop()
		<-done
	}
}

// Check fires every enabled automation due this minute that has not
// fired in it yet, one after another so announcements never overlap.
func (s *Scheduler) Check(ctx context.Context) {
	now := s.Now()
	minute := now.Truncate(time.Minute).Format(time.RFC3339)
	var due []Automation
	s.mu.Lock()
	for i := range s.list {
		a := &s.list[i]
		if a.Enabled && a.LastRun != minute && cronMatches(a.Schedule, now) {
			a.LastRun = minute
			due = append(due, *a)
		}
	}
	if len(due) > 0 {
		s.save()
	}
	s.mu.Unlock()
	for _, a := range due {
		s.Fire(ctx, a)
	}
}

// Fire asks the agent and announces a non-empty answer.
func (s *Scheduler) Fire(ctx context.Context, a Automation) {
	slog.Info("Automation firing", "name", a.Name)
	answer, err := s.ask(ctx, a.Prompt)
	if err != nil {
		slog.Error("Automation failed", "name", a.Name, "err", err)
		return
	}
	if answer = strings.TrimSpace(answer); answer == "" {
		slog.Warn("Automation had nothing to say", "name", a.Name)
		return
	}
	s.announce(answer)
	slog.Info("Automation announced", "name", a.Name)
}
