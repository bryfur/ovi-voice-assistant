package pipeline

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func newVA(s *fakeSTT, a *fakeAgent) *VoiceAssistant {
	return &VoiceAssistant{STT: s, TTS: &slowTTS{}, Agent: a}
}

func TestRunFullPipeline(t *testing.T) {
	ag := &fakeAgent{response: "Hello there friend."}
	out := &recordingOutput{}

	followUp := newVA(&fakeSTT{transcript: "hi", speech: true}, ag).Run(ctx, out, nil, nil)

	if followUp || out.eventNames() != "VAD_START,MIC_STOP,TTS_START,TTS_END" {
		t.Fatalf("followUp=%v events=%s", followUp, out.eventNames())
	}
	if p := out.played(); len(p) != 1 || p[0] != "Hello there friend." || ag.inputs[0] != "hi" {
		t.Fatalf("played=%v inputs=%v", p, ag.inputs)
	}
}

func TestRunNoSpeech(t *testing.T) {
	ag := &fakeAgent{response: "x"}
	out := &recordingOutput{}

	followUp := newVA(&fakeSTT{}, ag).Run(ctx, out, nil, nil)

	if followUp || len(ag.inputs) != 0 || out.eventNames() != "MIC_STOP,ERROR" {
		t.Fatalf("followUp=%v inputs=%v events=%s", followUp, ag.inputs, out.eventNames())
	}
}

func TestRunFollowUpListen(t *testing.T) {
	out := &recordingOutput{}

	followUp := newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{response: "Want more details? [LISTEN]"}).Run(ctx, out, nil, nil)

	if !followUp || out.eventNames() != "MIC_STOP,TTS_START,TTS_END,CONTINUE" || out.played()[0] != "Want more details?" {
		t.Fatalf("followUp=%v events=%s played=%v", followUp, out.eventNames(), out.played())
	}
}

func TestRunErrorsReachTheDevice(t *testing.T) {
	sttErr, agentErr, ttsErr := &recordingOutput{}, &recordingOutput{}, &recordingOutput{}

	newVA(&fakeSTT{err: errors.New("boom")}, &fakeAgent{}).Run(ctx, sttErr, nil, nil)
	newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{err: errors.New("boom")}).Run(ctx, agentErr, nil, nil)
	va := newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{response: "A long enough sentence. And more words here."})
	va.TTS = &slowTTS{err: errors.New("no voice")}
	va.Run(ctx, ttsErr, nil, nil)

	if sttErr.eventNames() != "ERROR" || agentErr.eventNames() != "MIC_STOP,TTS_START,ERROR" || ttsErr.eventNames() != "MIC_STOP,TTS_START,ERROR" {
		t.Fatalf("stt=%s agent=%s tts=%s", sttErr.eventNames(), agentErr.eventNames(), ttsErr.eventNames())
	}
}

func TestRunCancelledIsSilent(t *testing.T) {
	out := &recordingOutput{}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	followUp := newVA(&fakeSTT{err: context.Canceled}, &fakeAgent{}).Run(cancelled, out, nil, nil)

	if followUp || len(out.events) != 0 {
		t.Fatalf("followUp=%v events=%s", followUp, out.eventNames())
	}
}

func TestAnnounce(t *testing.T) {
	out := &recordingOutput{}

	newVA(&fakeSTT{}, &fakeAgent{}).Announce(ctx, out, "Timer done.")

	if out.eventNames() != "TTS_START,TTS_END" || out.played()[0] != "Timer done." {
		t.Fatalf("events=%s played=%v", out.eventNames(), out.played())
	}
}

// Audio for the first chunk must reach the device long before the agent
// has finished streaming (~1.3 s here).
func TestAudioStartsBeforeAgentFinishes(t *testing.T) {
	out := &recordingOutput{start: time.Now()}
	ag := &fakeAgent{response: "This is the first sentence. And here is the second one. Finally the third.", slow: 100 * time.Millisecond}

	newVA(&fakeSTT{transcript: "hi"}, ag).Run(ctx, out, nil, nil)

	if len(out.played()) < 3 {
		t.Fatalf("played = %v", out.played())
	}
	firstAudio := out.times[2] // after MIC_STOP and TTS_START
	t.Logf("first audio at %v, last event at %v", firstAudio.Round(time.Millisecond), out.times[len(out.times)-1].Round(time.Millisecond))
	if firstAudio > 700*time.Millisecond {
		t.Fatalf("first audio only after %v", firstAudio)
	}
}

func TestStopClosesProviders(t *testing.T) {
	s := &fakeSTT{}
	va := newVA(s, &fakeAgent{})

	va.Stop()

	if !s.closed {
		t.Fatal("STT not closed")
	}
}

func TestNewRejectsUnknownProviders(t *testing.T) {
	for _, mutate := range []func(*config.Settings){
		func(s *config.Settings) { s.STT.Provider = "bogus" },
		func(s *config.Settings) { s.TTS.Provider = "bogus" },
		func(s *config.Settings) { s.Transport.Codec = "bogus" },
	} {
		s := config.Default()
		mutate(s)
		if _, err := New(s); err == nil {
			t.Fatal("expected error")
		}
	}
	if va, err := New(config.Default()); err != nil || va.Rate() != 24000 {
		t.Fatalf("va=%v err=%v", va, err)
	}
}
