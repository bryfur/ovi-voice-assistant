// Package agent is the conversational core: an OpenAI tool-calling loop
// with built-in tools, MCP servers and sub-agents, streaming its answer
// token by token.
package agent

import (
	"fmt"
	"log/slog"
	"maps"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// Env is what tools can reach: the device that is listening, the shared
// music player and the scheduler. One exists per device.
type Env struct {
	Announce  func(text string) // speak on the device without waiting
	Music     *music.Player
	Scheduler *scheduler.Scheduler

	mu     sync.Mutex
	timers map[string]timer
}

type timer struct {
	stop *time.Timer
	due  time.Time
}

// SetTimer announces "Your <label> timer is done." after d, replacing any
// timer with the same label.
func (e *Env) SetTimer(d time.Duration, label string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if old, ok := e.timers[label]; ok {
		old.stop.Stop()
	}
	if e.timers == nil {
		e.timers = map[string]timer{}
	}
	e.timers[label] = timer{due: time.Now().Add(d), stop: time.AfterFunc(d, func() {
		e.mu.Lock()
		delete(e.timers, label)
		e.mu.Unlock()
		slog.Info("Timer fired", "label", label)
		if e.Announce != nil {
			e.Announce(fmt.Sprintf("Your %s timer is done.", label))
		}
	})}
	slog.Info("Timer set", "label", label, "duration", d)
}

// CancelTimer stops a timer; false if there is none with that label.
func (e *Env) CancelTimer(label string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	t, ok := e.timers[label]
	if ok {
		t.stop.Stop()
		delete(e.timers, label)
	}
	return ok
}

// Timers is the time left on each running timer.
func (e *Env) Timers() map[string]time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	left := make(map[string]time.Duration, len(e.timers))
	for label, t := range maps.All(e.timers) {
		left[label] = max(0, time.Until(t.due))
	}
	return left
}
