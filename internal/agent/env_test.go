package agent

import (
	"testing"
	"time"
)

func TestTimersSetCancelAndReplace(t *testing.T) {
	e := &Env{}

	e.SetTimer(time.Minute, "pasta")
	e.SetTimer(100*time.Second, "a")
	e.SetTimer(5*time.Second, "a") // replaces
	left := e.Timers()

	if len(left) != 2 || left["pasta"] < 58*time.Second || left["pasta"] > time.Minute || left["a"] > 5*time.Second {
		t.Fatalf("timers = %v", left)
	}
	if !e.CancelTimer("pasta") || e.CancelTimer("pasta") || !e.CancelTimer("a") || len(e.Timers()) != 0 {
		t.Fatal("cancel semantics wrong")
	}
}

func TestTimerAnnouncesWhenDone(t *testing.T) {
	announced := make(chan string, 1)
	e := &Env{Announce: func(text string) { announced <- text }}

	e.SetTimer(10*time.Millisecond, "egg")

	select {
	case text := <-announced:
		if text != "Your egg timer is done." || len(e.Timers()) != 0 {
			t.Fatalf("announced=%q timers=%v", text, e.Timers())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timer did not fire")
	}
}
