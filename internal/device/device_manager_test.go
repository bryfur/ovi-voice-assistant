package device

import (
	"context"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

func newManager(t *testing.T, hosts ...string) (*DeviceManager, []*fakeTransport, *fakePipeline) {
	t.Helper()
	var devices []config.DeviceConfig
	for _, h := range hosts {
		devices = append(devices, config.DeviceConfig{Host: h, Port: 6055})
	}
	var transports []*fakeTransport
	factory := func(dev config.DeviceConfig) transport.DeviceTransport {
		tr := &fakeTransport{}
		transports = append(transports, tr)
		return tr
	}
	s := config.Default()
	s.Transport.Codec = "pcm"
	pl := newFakePipeline()
	m, err := NewDeviceManager(devices, s, pl, 16000, factory)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range m.Connections() {
		c.SetupDelay = 0
	}
	m.Window = 20 * time.Millisecond
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Stop(context.Background()) })
	return m, transports, pl
}

func TestSingleDeviceNoArbitration(t *testing.T) {
	m, transports, pl := newManager(t, "a")
	pl.block = true

	transports[0].eventCB(transport.EventWakeWord, []byte("w"))
	<-pl.started

	if m.MultiDevice() || m.PendingCandidates() != 0 || m.Connections()[0].onWake != nil {
		t.Fatal("single device should start immediately without arbitration")
	}
}

func TestMultiDeviceArbitrationHighestScoreWins(t *testing.T) {
	m, transports, pl := newManager(t, "a", "b", "c")
	pl.block = true
	payload := func(peak, ambient uint16) []byte {
		p := make([]byte, 4)
		p[0], p[1] = byte(peak), byte(peak>>8)
		p[2], p[3] = byte(ambient), byte(ambient>>8)
		return append(p, 'w')
	}

	transports[0].eventCB(transport.EventWakeWord, payload(1000, 100)) // 10000
	transports[1].eventCB(transport.EventWakeWord, payload(3000, 100)) // 30000 → winner
	transports[2].eventCB(transport.EventWakeWord, payload(500, 100))  // 5000
	pending := m.PendingCandidates()
	<-pl.started
	waitFor(t, func() bool {
		return transports[0].hasEvent(transport.EventWakeAbort) && transports[2].hasEvent(transport.EventWakeAbort)
	})

	if pending != 3 || pl.runCount() != 1 || !m.MultiDevice() {
		t.Fatalf("pending=%d runs=%d", pending, pl.runCount())
	}
	if transports[1].hasEvent(transport.EventWakeAbort) || !m.Connections()[1].TaskRunning() {
		t.Fatal("winner must not be aborted")
	}
}

func TestSecondCandidateDoesNotRestartTimer(t *testing.T) {
	m, _, _ := newManager(t, "a", "b")
	m.Window = time.Hour
	conns := m.Connections()

	m.onWake(conns[0], 1, "w")
	first := m.timer
	m.onWake(conns[1], 2, "w")

	if first == nil || m.timer != first || m.PendingCandidates() != 2 {
		t.Fatal("timer should be created once per window")
	}
	m.Stop(context.Background())
	if m.timer != nil {
		t.Fatal("Stop should cancel the timer")
	}
}

func TestResolveArbitrationEmptyIsNoop(t *testing.T) {
	m, _, pl := newManager(t, "a", "b")

	m.ResolveArbitration()

	if pl.runCount() != 0 {
		t.Fatal("nothing should run")
	}
}

func TestAnnounceAllAndSetters(t *testing.T) {
	m, _, pl := newManager(t, "a", "b")

	m.AnnounceAll(context.Background(), "hello")
	waitFor(t, func() bool {
		pl.mu.Lock()
		defer pl.mu.Unlock()
		return len(pl.announces) == 2
	})
	m.SetScheduler(nil)
}
