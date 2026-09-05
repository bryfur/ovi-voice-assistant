package pipeline

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

// slowAgent streams one word every 100 ms.
type slowAgent struct{ fakeAgent }

func (s *slowAgent) RunStreamed(_ context.Context, _ string, _ *agent.Context, onToken func(string)) error {
	for _, w := range strings.SplitAfter("This is the first sentence. And here is the second one. Finally the third.", " ") {
		onToken(w)
		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

type timedOutput struct {
	recordingOutput
	mu    sync.Mutex
	times []time.Duration
	start time.Time
}

func (o *timedOutput) SendAudio(ctx context.Context, pcm []byte) error {
	o.mu.Lock()
	o.times = append(o.times, time.Since(o.start))
	o.mu.Unlock()
	return o.recordingOutput.SendAudio(ctx, pcm)
}

func (o *timedOutput) SendEvent(ctx context.Context, e transport.EventType, p []byte) error {
	o.mu.Lock()
	o.times = append(o.times, time.Since(o.start))
	o.mu.Unlock()
	return o.recordingOutput.SendEvent(ctx, e, p)
}

// Audio for the first chunk must reach the output long before the agent
// has finished streaming (~1.3 s here).
func TestAudioStartsBeforeAgentFinishes(t *testing.T) {
	out := &timedOutput{start: time.Now()}
	va := &VoiceAssistant{STT: &fakeSTT{transcript: "hi"}, TTS: &slowTTS{}, Agent: &slowAgent{}}

	va.Run(context.Background(), out, nil, nil)

	firstAudio := time.Duration(-1)
	for i, e := range out.recordingOutput.events {
		_ = i
		_ = e
	}
	// events and audio share one timeline: TTS_START, then audio chunks...
	if len(out.played()) < 3 {
		t.Fatalf("played = %v", out.played())
	}
	// times[0]=MIC_STOP, times[1]=TTS_START, times[2]=first audio
	firstAudio = out.times[2]
	t.Logf("first audio at %v; chunks=%v; last event at %v", firstAudio.Round(time.Millisecond), out.played(), out.times[len(out.times)-1].Round(time.Millisecond))
	if firstAudio > 700*time.Millisecond {
		t.Fatalf("first audio only after %v", firstAudio)
	}
}
