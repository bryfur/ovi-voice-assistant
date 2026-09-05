package pipeline

import (
	"context"
	"testing"
	"time"
)

func TestSubmitPlaysText(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{}, out)

	err := <-q.Submit("Hello world.")
	q.Stop()

	if err != nil || len(out.played()) != 1 || out.played()[0] != "Hello world." {
		t.Fatalf("err=%v played=%v", err, out.played())
	}
}

func TestSubmitReturnsBeforePlayback(t *testing.T) {
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 50 * time.Millisecond}, &recordingOutput{})

	start := time.Now()
	done := q.Submit("Slow sentence here.")
	elapsed := time.Since(start)
	<-done
	q.Stop()

	if elapsed > 40*time.Millisecond {
		t.Fatalf("Submit blocked for %v", elapsed)
	}
}

func TestSubmissionsPlayInOrder(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 5 * time.Millisecond}, out)

	q.Submit("First sentence here.")
	q.Submit("Second sentence here.")
	q.Submit("Third sentence here.")
	q.Stop()

	if got := out.played(); len(got) != 3 || got[0] != "First sentence here." || got[2] != "Third sentence here." {
		t.Fatalf("played = %v", got)
	}
}

func TestStopDrainsAndRejectsLater(t *testing.T) {
	out := &recordingOutput{}
	q := NewSpeechQueue(context.Background(), &slowTTS{delay: 10 * time.Millisecond}, out)
	q.Submit("Something to say here.")

	q.Stop()
	q.Stop()

	if len(out.played()) != 1 {
		t.Fatal("Stop must wait for queued speech")
	}
	if err := <-q.Submit("after stop"); err == nil {
		t.Fatal("submissions after Stop should fail")
	}
}
