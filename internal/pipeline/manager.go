package pipeline

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// arbitration is how long to wait for competing wake words before the
// best-placed device answers.
const arbitration = 500 * time.Millisecond

type candidate struct {
	conn  *Connection
	score int
	word  string
}

// Manager runs a Connection per device. With several devices, a wake
// word heard by more than one of them is answered only by the one that
// heard it best.
type Manager struct {
	Connections []*Connection
	Window      time.Duration

	mu         sync.Mutex
	candidates []candidate
	timer      *time.Timer
}

// NewManager prepares a connection for each transport.
func NewManager(transports []device.Transport, codecName string, v Voice, player *music.Player) (*Manager, error) {
	m := &Manager{Window: arbitration}
	for _, t := range transports {
		c, err := NewConnection(t, codecName, v, player)
		if err != nil {
			return nil, err
		}
		if len(transports) > 1 {
			c.OnWake = m.wake
		}
		m.Connections = append(m.Connections, c)
	}
	return m, nil
}

// Start connects every device.
func (m *Manager) Start() error {
	for _, c := range m.Connections {
		if err := c.Start(); err != nil {
			return fmt.Errorf("device %s: %w", c.Name, err)
		}
	}
	slog.Info("Devices connected", "count", len(m.Connections), "arbitration", len(m.Connections) > 1)
	return nil
}

// Stop disconnects every device.
func (m *Manager) Stop() {
	m.mu.Lock()
	if m.timer != nil {
		m.timer.Stop()
		m.timer = nil
	}
	m.mu.Unlock()
	for _, c := range m.Connections {
		if err := c.Stop(); err != nil {
			slog.Debug("Device stop failed", "device", c.Name, "err", err)
		}
	}
}

// Announce speaks text on every device.
func (m *Manager) Announce(text string) {
	for _, c := range m.Connections {
		c.Announce(text)
	}
}

// SetScheduler makes the automation tools available on every device.
func (m *Manager) SetScheduler(s *scheduler.Scheduler) {
	for _, c := range m.Connections {
		c.SetScheduler(s)
	}
}

// wake collects a candidate; the first one opens the window.
func (m *Manager) wake(c *Connection, score int, word string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.candidates = append(m.candidates, candidate{c, score, word})
	slog.Info("Wake candidate", "device", c.Name, "score", score, "in_window", len(m.candidates))
	if len(m.candidates) == 1 {
		m.timer = time.AfterFunc(m.Window, m.resolve)
	}
}

// resolve starts a session on the best candidate and stands the rest down.
func (m *Manager) resolve() {
	m.mu.Lock()
	m.timer = nil
	cands := m.candidates
	m.candidates = nil
	m.mu.Unlock()
	if len(cands) == 0 {
		return
	}
	winner := slices.MaxFunc(cands, func(a, b candidate) int { return cmp.Compare(a.score, b.score) })
	slog.Info("Wake arbitration", "winner", winner.conn.Name, "score", winner.score, "candidates", len(cands))
	for _, c := range cands {
		if c.conn != winner.conn {
			c.conn.AbortWake()
		}
	}
	winner.conn.StartSession(winner.word)
}
