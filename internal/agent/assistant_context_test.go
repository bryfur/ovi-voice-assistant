package agent

import (
	"sync"
	"testing"
	"time"
)

func TestScheduleAndCancelTimer(t *testing.T) {
	c := &Context{}

	c.ScheduleTimer(60, "pasta")
	status := c.TimerStatus()
	cancelled := c.CancelTimer("pasta")

	if len(status) != 1 || status["pasta"] < 58 || status["pasta"] > 60 {
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

	c.ScheduleTimer(0.01, "egg")

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

func TestTimerWithoutAnnounceDoesNotPanic(t *testing.T) {
	c := &Context{}

	c.ScheduleTimer(0.001, "x")
	time.Sleep(20 * time.Millisecond)
}

func TestRescheduleReplacesTimer(t *testing.T) {
	c := &Context{}

	c.ScheduleTimer(100, "a")
	c.ScheduleTimer(5, "a")

	if s := c.TimerStatus(); len(s) != 1 || s["a"] > 5 {
		t.Fatalf("status = %v", s)
	}
}
