package pipeline

import (
	"context"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// silentService plays a track by blocking until it is interrupted.
type silentService struct{ started chan struct{} }

func (s *silentService) Search(context.Context, string, int) ([]music.Track, error) { return nil, nil }
func (s *silentService) Play(ctx context.Context, _ music.Track, _ device.Output) error {
	s.started <- struct{}{}
	<-ctx.Done()
	return ctx.Err()
}

func newManager(t *testing.T, n int) (*Manager, []*fakeTransport, *fakeVoice) {
	t.Helper()
	var transports []device.Transport
	var fakes []*fakeTransport
	for range n {
		tr := &fakeTransport{}
		transports, fakes = append(transports, tr), append(fakes, tr)
	}
	v := newFakeVoice()
	m, err := NewManager(transports, "pcm", v, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range m.Connections {
		c.SetupDelay = 0
	}
	m.Window = 20 * time.Millisecond
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Stop)
	return m, fakes, v
}

func TestSingleDeviceNeedsNoArbitration(t *testing.T) {
	m, transports, v := newManager(t, 1)
	v.block = true

	transports[0].h.Event(device.EventWakeWord, []byte("w"))
	<-v.started

	if m.Connections[0].OnWake != nil || len(m.candidates) != 0 {
		t.Fatal("single device should start immediately")
	}
}

func TestHighestScoreWinsArbitration(t *testing.T) {
	m, transports, v := newManager(t, 3)
	v.block = true

	transports[0].h.Event(device.EventWakeWord, wakePayload(1000, 100, "w")) // 10000
	transports[1].h.Event(device.EventWakeWord, wakePayload(3000, 100, "w")) // 30000: winner
	transports[2].h.Event(device.EventWakeWord, wakePayload(500, 100, "w"))  // 5000
	m.mu.Lock()
	pending := len(m.candidates)
	m.mu.Unlock()
	<-v.started
	waitFor(t, func() bool {
		return transports[0].has(device.EventWakeAbort) && transports[2].has(device.EventWakeAbort)
	})

	runs, _, _ := v.count()
	if pending != 3 || runs != 1 || transports[1].has(device.EventWakeAbort) || !m.Connections[1].Busy() {
		t.Fatalf("pending=%d runs=%d", pending, runs)
	}
}

func TestWindowOpensOnceAndStopCancelsIt(t *testing.T) {
	m, _, _ := newManager(t, 2)
	m.Window = time.Hour

	m.wake(m.Connections[0], 1, "w")
	first := m.timer
	m.wake(m.Connections[1], 2, "w")
	same := m.timer == first
	m.Stop()

	if first == nil || !same || m.timer != nil {
		t.Fatal("one timer per window, cleared by Stop")
	}
	m.resolve() // nothing pending after Stop cleared the timer, candidates remain harmless
}

func TestAnnounceReachesEveryDevice(t *testing.T) {
	m, _, v := newManager(t, 2)

	m.Announce("hello")
	waitFor(t, func() bool {
		v.mu.Lock()
		defer v.mu.Unlock()
		return len(v.announces) == 2
	})
	m.SetScheduler(nil)
}
