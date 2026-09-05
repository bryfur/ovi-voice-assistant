// Package agent implements the conversational agent: an OpenAI-compatible
// tool-calling loop with built-in tools, MCP servers and sub-agents.
package agent

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/memory"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"github.com/bryfur/ovi-voice-assistant/internal/scheduler"
)

// MemoryStore is the memory surface used by the agent.
type MemoryStore interface {
	Recall(ctx context.Context, query string, budget memory.Budget, maxTokens int) (memory.RecallResult, error)
	Retain(ctx context.Context, content, extraContext string) (memory.RetainResult, error)
}

// Context is the run context available to all agent tools during a
// pipeline run. One is created per device connection.
type Context struct {
	// Announce triggers a TTS announcement on the device (fire-and-forget).
	Announce func(text string)
	// Say speaks text immediately during a pipeline run without interrupting it.
	Say func(ctx context.Context, text string) error
	// MusicPlayer is the per-device music player.
	MusicPlayer *music.MusicPlayer
	// MusicGroup is the shared group for synchronized multi-device playback.
	MusicGroup *music.MusicGroup
	// Scheduler creates/manages proactive automations.
	Scheduler *scheduler.Scheduler
	// Memory is persistent memory for fact extraction and recall.
	Memory MemoryStore

	mu     sync.Mutex
	timers map[string]*timerEntry
}

type timerEntry struct {
	timer    *time.Timer
	fireTime time.Time
}

// ScheduleTimer schedules a timer that announces on the device when it fires.
func (c *Context) ScheduleTimer(seconds float64, label string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.timers == nil {
		c.timers = map[string]*timerEntry{}
	}
	if old, ok := c.timers[label]; ok {
		old.timer.Stop()
	}
	d := time.Duration(seconds * float64(time.Second))
	entry := &timerEntry{fireTime: time.Now().Add(d)}
	entry.timer = time.AfterFunc(d, func() { c.onTimerFire(label) })
	c.timers[label] = entry
	slog.Info("Timer scheduled", "label", label, "seconds", int(seconds))
}

// CancelTimer cancels a named timer. Returns true if found.
func (c *Context) CancelTimer(label string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.timers[label]
	if !ok {
		return false
	}
	entry.timer.Stop()
	delete(c.timers, label)
	slog.Info("Timer cancelled", "label", label)
	return true
}

// TimerStatus returns remaining seconds for each active timer.
func (c *Context) TimerStatus() map[string]float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	out := make(map[string]float64, len(c.timers))
	for label, e := range c.timers {
		remaining := e.fireTime.Sub(now).Seconds()
		if remaining < 0 {
			remaining = 0
		}
		out[label] = remaining
	}
	return out
}

// onTimerFire announces timer completion on the device.
func (c *Context) onTimerFire(label string) {
	c.mu.Lock()
	delete(c.timers, label)
	announce := c.Announce
	c.mu.Unlock()
	text := fmt.Sprintf("Your %s timer is done.", label)
	slog.Info("Timer fired", "label", label)
	if announce != nil {
		announce(text)
	} else {
		slog.Warn("No announce callback — cannot notify device")
	}
}
