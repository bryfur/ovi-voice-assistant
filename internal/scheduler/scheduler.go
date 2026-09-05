// Package scheduler runs persistent cron-based automations that send
// agent prompts proactively and announce the result.
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
	"strings"
	"sync"
	"time"
)

// CheckInterval is how often the scheduler checks for automations to fire.
const CheckInterval = 30 * time.Second

// Automation is a single scheduled automation.
type Automation struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Schedule string `json:"schedule"` // cron expression: minute hour dom month dow
	Prompt   string `json:"prompt"`   // what to ask the agent
	Enabled  bool   `json:"enabled"`
	LastRun  string `json:"last_run"` // RFC3339 — prevents double-firing within same minute
}

// RunPrompt runs a prompt through the agent and returns the response.
type RunPrompt func(ctx context.Context, prompt string) (string, error)

// Announce speaks text on all devices.
type Announce func(ctx context.Context, text string) error

// Scheduler runs persistent cron automations in the background.
type Scheduler struct {
	path      string
	runPrompt RunPrompt
	announce  Announce

	mu          sync.Mutex
	automations []Automation
	cancel      context.CancelFunc
	loopDone    chan struct{}
	fires       sync.WaitGroup

	// Now is the clock; tests may override it.
	Now func() time.Time
}

// New creates a scheduler persisting to path.
func New(path string, runPrompt RunPrompt, announce Announce) *Scheduler {
	return &Scheduler{path: path, runPrompt: runPrompt, announce: announce, Now: time.Now}
}

// -- Persistence --

// Load reads automations from disk. A missing or corrupt file yields an
// empty list.
func (s *Scheduler) Load() {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(s.path)
	if err != nil {
		s.automations = nil
		return
	}
	var autos []Automation
	if err := json.Unmarshal(data, &autos); err != nil {
		slog.Error("Failed to load automations", "path", s.path, "err", err)
		s.automations = nil
		return
	}
	s.automations = autos
	slog.Info("Loaded automations", "count", len(autos), "path", s.path)
}

// save persists automations. Caller must hold mu.
func (s *Scheduler) save() {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		slog.Error("Failed to create automations dir", "err", err)
		return
	}
	autos := s.automations
	if autos == nil {
		autos = []Automation{}
	}
	data, err := json.MarshalIndent(autos, "", "  ")
	if err != nil {
		return
	}
	if err := os.WriteFile(s.path, data, 0o644); err != nil {
		slog.Error("Failed to save automations", "err", err)
	}
}

// -- Public API (called by agent tools) --

// Automations returns a copy of the current list.
func (s *Scheduler) Automations() []Automation {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Automation(nil), s.automations...)
}

// Create validates, persists and returns a new automation.
func (s *Scheduler) Create(name, schedule, prompt string) (Automation, error) {
	if !ValidateCron(schedule) {
		return Automation{}, fmt.Errorf(
			"invalid cron expression %q: expected 5 fields (minute hour day-of-month month day-of-week)",
			schedule)
	}
	auto := Automation{
		ID:       newID(),
		Name:     name,
		Schedule: schedule,
		Prompt:   prompt,
		Enabled:  true,
	}
	s.mu.Lock()
	s.automations = append(s.automations, auto)
	s.save()
	s.mu.Unlock()
	slog.Info("Automation created", "name", name, "schedule", schedule, "prompt", truncate(prompt, 60))
	return auto, nil
}

// Delete removes an automation by name. Returns true if found.
func (s *Scheduler) Delete(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.automations[:0:0]
	for _, a := range s.automations {
		if a.Name != name {
			kept = append(kept, a)
		}
	}
	if len(kept) == len(s.automations) {
		return false
	}
	s.automations = kept
	s.save()
	slog.Info("Automation deleted", "name", name)
	return true
}

// SetEnabled enables or disables an automation by name. Returns true if found.
func (s *Scheduler) SetEnabled(name string, enabled bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.automations {
		if s.automations[i].Name == name {
			s.automations[i].Enabled = enabled
			s.save()
			slog.Info("Automation toggled", "name", name, "enabled", enabled)
			return true
		}
	}
	return false
}

// -- Background loop --

// Start launches the periodic check loop.
func (s *Scheduler) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	s.loopDone = make(chan struct{})
	go s.loop(ctx, s.loopDone)
	slog.Info("Scheduler started", "automations", len(s.automations))
}

// Stop halts the loop and waits for in-flight automations to finish.
func (s *Scheduler) Stop() {
	s.mu.Lock()
	cancel := s.cancel
	done := s.loopDone
	s.cancel = nil
	s.loopDone = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
	s.fires.Wait()
	slog.Info("Scheduler stopped")
}

func (s *Scheduler) loop(ctx context.Context, done chan struct{}) {
	defer close(done)
	ticker := time.NewTicker(CheckInterval)
	defer ticker.Stop()
	for {
		s.Check(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Check fires every enabled automation whose schedule matches the current
// minute and has not already fired in that minute.
func (s *Scheduler) Check(ctx context.Context) {
	now := s.Now()
	nowMinute := now.Truncate(time.Minute)

	s.mu.Lock()
	var toFire []Automation
	for i := range s.automations {
		a := &s.automations[i]
		if !a.Enabled || !CronMatches(a.Schedule, now) {
			continue
		}
		if a.LastRun != "" {
			if last, err := time.Parse(time.RFC3339Nano, a.LastRun); err == nil &&
				!last.Truncate(time.Minute).Before(nowMinute) {
				continue
			}
		}
		a.LastRun = now.Format(time.RFC3339Nano)
		toFire = append(toFire, *a)
	}
	if len(toFire) > 0 {
		s.save()
	}
	s.mu.Unlock()

	for _, a := range toFire {
		s.fires.Add(1)
		go func(a Automation) {
			defer s.fires.Done()
			s.Fire(ctx, a)
		}(a)
	}
}

// Fire runs the automation's prompt through the agent and announces the result.
func (s *Scheduler) Fire(ctx context.Context, a Automation) {
	slog.Info("Automation firing", "name", a.Name, "prompt", truncate(a.Prompt, 60))
	response, err := s.runPrompt(ctx, a.Prompt)
	if err != nil {
		slog.Error("Automation failed", "name", a.Name, "err", err)
		return
	}
	if strings.TrimSpace(response) == "" {
		slog.Warn("Automation produced empty response", "name", a.Name)
		return
	}
	if err := s.announce(ctx, response); err != nil {
		slog.Error("Automation announce failed", "name", a.Name, "err", err)
		return
	}
	slog.Info("Automation announced", "name", a.Name, "response", truncate(response, 80))
}

func newID() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
