package agent

import (
	"sync"
	"testing"
	"time"
)

func TestScheduleAndCancelTimer(t *testing.T) {
	c := &Context{}

	c.ScheduleTimer(time.Minute, "pasta")
	status := c.TimerStatus()
	cancelled := c.CancelTimer("pasta")

	if len(status) != 1 || status["pasta"] < 58*time.Second || status["pasta"] > time.Minute {
		t.Fatalf("status = %v", status)
	}
	if !cancelled || c.CancelTimer("pasta") || len(c.TimerStatus()) != 0 {
		t.Fatal("cancel semantics wrong")
	}
}

func TestTimerFiresAnnouncement(t *testing.T) {
	var mu sync.Mutex
	var announced string
	done := make(chan struct{})
	c := &Context{Announce: func(text string) {
		mu.Lock()
		announced = text
		mu.Unlock()
		close(done)
	}}

	c.ScheduleTimer(10*time.Millisecond, "egg")

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timer did not fire")
	}
	mu.Lock()
	defer mu.Unlock()
	if announced != "Your egg timer is done." || len(c.TimerStatus()) != 0 {
		t.Fatalf("announced=%q status=%v", announced, c.TimerStatus())
	}
}

func TestRescheduleReplacesTimer(t *testing.T) {
	c := &Context{}

	c.ScheduleTimer(100*time.Second, "a")
	c.ScheduleTimer(5*time.Second, "a")

	if s := c.TimerStatus(); len(s) != 1 || s["a"] > 5*time.Second {
		t.Fatalf("status = %v", s)
	}
	c.CancelTimer("a")
}
