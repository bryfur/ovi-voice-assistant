// Package agent implements the conversational agent: an OpenAI tool-calling
// loop with built-in tools, MCP servers and sub-agents.
package agent

import (
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"github.com/bryfur/ovi-voice-assistant/internal/scheduler"
)

// Context is what tools can reach during a run. One exists per device.
type Context struct {
	Announce    func(text string) // speak on the device (fire-and-forget)
	MusicPlayer *music.MusicPlayer
	MusicGroup  *music.MusicGroup
	Scheduler   *scheduler.Scheduler

	mu     sync.Mutex
	timers map[string]*timer
}

type timer struct {
	t   *time.Timer
	due time.Time
}

// ScheduleTimer announces "Your <label> timer is done." after d.
func (c *Context) ScheduleTimer(d time.Duration, label string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.timers == nil {
		c.timers = map[string]*timer{}
	}
	if old := c.timers[label]; old != nil {
		old.t.Stop()
	}
	c.timers[label] = &timer{
		due: time.Now().Add(d),
		t: time.AfterFunc(d, func() {
			c.mu.Lock()
			delete(c.timers, label)
			announce := c.Announce
			c.mu.Unlock()
			slog.Info("Timer fired", "label", label)
			if announce != nil {
				announce(fmt.Sprintf("Your %s timer is done.", label))
			}
		}),
	}
	slog.Info("Timer scheduled", "label", label, "duration", d)
}

// CancelTimer stops a timer; false if none by that label.
func (c *Context) CancelTimer(label string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.timers[label]
	if t == nil {
		return false
	}
	t.t.Stop()
	delete(c.timers, label)
	return true
}

// TimerStatus returns the remaining time of each active timer.
func (c *Context) TimerStatus() map[string]time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]time.Duration, len(c.timers))
	for label, t := range c.timers {
		out[label] = max(0, time.Until(t.due))
	}
	return out
}
