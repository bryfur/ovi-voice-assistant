package pipeline

import (
	"context"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// arbitrationWindow is how long to wait for competing wake events before
// picking a winner.
const arbitrationWindow = 500 * time.Millisecond

// wakeCandidate is a device that reported a wake word in the current window.
type wakeCandidate struct {
	conn     *DeviceConnection
	score    int
	wakeWord string
}

// TransportFactory builds a transport for a device; tests may override it.
type TransportFactory func(dev config.DeviceConfig) device.Transport

// DeviceManager creates and manages DeviceConnection instances for WiFi
// devices and arbitrates competing wake words.
type DeviceManager struct {
	connections []*DeviceConnection
	multiDevice bool
	musicGroup  *music.MusicGroup

	mu         sync.Mutex
	candidates []wakeCandidate
	timer      *time.Timer

	// Window is the arbitration window.
	Window time.Duration
}

// NewDeviceManager builds connections for the configured devices.
func NewDeviceManager(devices []config.DeviceConfig, settings *config.Settings, p Runner, ttsRate int, factory TransportFactory) (*DeviceManager, error) {
	if factory == nil {
		factory = func(dev config.DeviceConfig) device.Transport {
			return device.NewWiFiTransport(dev.Host, dev.Port, dev.EncryptionKey)
		}
	}
	m := &DeviceManager{multiDevice: len(devices) > 1, Window: arbitrationWindow}

	// Shared music group for synchronized multi-device playback. Created
	// here so all devices share the same player/queue.
	musicCodec, err := codec.NewCodec(settings.Transport.Codec, 48000, 2, codec.LC3MusicNByte)
	if err != nil {
		return nil, fmt.Errorf("music codec: %w", err)
	}
	m.musicGroup = music.NewMusicGroup(musicCodec.SampleRate(), musicCodec.Channels(), music.Browsers())

	// Only use the arbitration callback when there are multiple devices.
	var onWake WakeCallback
	if m.multiDevice {
		onWake = m.onWake
	}
	for _, dev := range devices {
		c, err := codec.NewCodec(settings.Transport.Codec, ttsRate, 1, 0)
		if err != nil {
			return nil, err
		}
		m.connections = append(m.connections, NewDeviceConnection(
			factory(dev), c, p, settings,
			DeviceOptions{OnWake: onWake, Name: dev.Host, MusicGroup: m.musicGroup},
		))
	}
	return m, nil
}

// Connections returns the managed connections.
func (m *DeviceManager) Connections() []*DeviceConnection { return m.connections }

// MultiDevice reports whether arbitration is enabled.
func (m *DeviceManager) MultiDevice() bool { return m.multiDevice }

// Start connects all devices.
func (m *DeviceManager) Start() error {
	for _, c := range m.connections {
		if err := c.Start(); err != nil {
			return fmt.Errorf("device %s: %w", c.Name, err)
		}
	}
	if m.multiDevice {
		slog.Info("Managing devices — multi-device arbitration enabled",
			"count", len(m.connections), "window", m.Window)
	} else {
		slog.Info("Managing 1 device")
	}
	return nil
}

// Stop disconnects all devices.
func (m *DeviceManager) Stop(ctx context.Context) {
	m.mu.Lock()
	if m.timer != nil {
		m.timer.Stop()
		m.timer = nil
	}
	m.mu.Unlock()
	m.musicGroup.Close(ctx)
	for _, c := range m.connections {
		if err := c.Stop(); err != nil {
			slog.Debug("Device stop error", "device", c.Name, "err", err)
		}
	}
}

// AnnounceAll announces text on every connected device.
func (m *DeviceManager) AnnounceAll(ctx context.Context, text string) error {
	for _, c := range m.connections {
		c.Announce(text)
	}
	return nil
}

// SetScheduler attaches the scheduler to all device connections.
func (m *DeviceManager) SetScheduler(s *scheduler.Scheduler) {
	for _, c := range m.connections {
		c.SetScheduler(s)
	}
}

// -- Wake-word arbitration --

// onWake is called by any DeviceConnection when a wake word is detected.
func (m *DeviceManager) onWake(conn *DeviceConnection, score int, wakeWord string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.candidates = append(m.candidates, wakeCandidate{conn, score, wakeWord})
	slog.Info("Wake candidate", "device", conn.Name, "score", score, "wake_word", wakeWord, "in_window", len(m.candidates))
	if len(m.candidates) == 1 {
		// First candidate — start the arbitration timer.
		m.timer = time.AfterFunc(m.Window, m.ResolveArbitration)
	}
}

// PendingCandidates returns the number of candidates in the current window.
func (m *DeviceManager) PendingCandidates() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.candidates)
}

// ResolveArbitration picks the best device and starts its pipeline; the
// rest are aborted.
func (m *DeviceManager) ResolveArbitration() {
	m.mu.Lock()
	m.timer = nil
	candidates := m.candidates
	m.candidates = nil
	m.mu.Unlock()
	if len(candidates) == 0 {
		return
	}
	// Sort by score descending — highest normalised energy wins.
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	winner := candidates[0]
	slog.Info("Wake arbitration resolved", "winner", winner.conn.Name, "score", winner.score, "candidates", len(candidates))
	winner.conn.StartPipeline(winner.wakeWord)
	for _, loser := range candidates[1:] {
		slog.Info("  loser", "device", loser.conn.Name, "score", loser.score)
		loser.conn.AbortWake()
	}
}
