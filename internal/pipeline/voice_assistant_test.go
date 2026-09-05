package pipeline

import (
	"context"
	"errors"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

func newVA(s *fakeSTT, a *fakeAgent) *VoiceAssistant {
	return &VoiceAssistant{STT: s, TTS: &slowTTS{}, Agent: a}
}

func TestRunFullPipeline(t *testing.T) {
	ag := &fakeAgent{response: "Hello there friend."}
	va := newVA(&fakeSTT{transcript: "hi", speech: true}, ag)
	out := &recordingOutput{}

	followUp := va.Run(context.Background(), out, nil, nil)

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

	followUp := newVA(&fakeSTT{}, ag).Run(context.Background(), out, nil, nil)

	if followUp || len(ag.inputs) != 0 || out.eventNames() != "MIC_STOP,ERROR" {
		t.Fatalf("followUp=%v inputs=%v events=%s", followUp, ag.inputs, out.eventNames())
	}
}

func TestRunFollowUpListen(t *testing.T) {
	out := &recordingOutput{}

	followUp := newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{response: "Want more details? [LISTEN]"}).Run(context.Background(), out, nil, nil)

	if !followUp || out.eventNames() != "MIC_STOP,TTS_START,TTS_END,CONTINUE" || out.played()[0] != "Want more details?" {
		t.Fatalf("followUp=%v events=%s played=%v", followUp, out.eventNames(), out.played())
	}
}

func TestRunErrorsSendPipelineError(t *testing.T) {
	sttErr := &recordingOutput{}
	agentErr := &recordingOutput{}

	newVA(&fakeSTT{err: errors.New("boom")}, &fakeAgent{}).Run(context.Background(), sttErr, nil, nil)
	newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{err: errors.New("boom")}).Run(context.Background(), agentErr, nil, nil)

	if sttErr.eventNames() != "ERROR" || agentErr.eventNames() != "MIC_STOP,TTS_START,ERROR" {
		t.Fatalf("stt=%s agent=%s", sttErr.eventNames(), agentErr.eventNames())
	}
}

func TestRunCancelledIsSilent(t *testing.T) {
	out := &recordingOutput{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	followUp := newVA(&fakeSTT{err: context.Canceled}, &fakeAgent{}).Run(ctx, out, nil, nil)

	if followUp || len(out.events) != 0 {
		t.Fatalf("followUp=%v events=%s", followUp, out.eventNames())
	}
}

func TestAnnounce(t *testing.T) {
	out := &recordingOutput{}

	newVA(&fakeSTT{}, &fakeAgent{}).Announce(context.Background(), out, "Timer done.")

	if out.eventNames() != "TTS_START,TTS_END" || out.played()[0] != "Timer done." {
		t.Fatalf("events=%s played=%v", out.eventNames(), out.played())
	}
}

func TestStopClosesProviders(t *testing.T) {
	s := &fakeSTT{}
	va := newVA(s, &fakeAgent{})

	va.Stop(context.Background())

	if !s.closed {
		t.Fatal("STT not closed")
	}
}

func TestNewRejectsUnknownProviders(t *testing.T) {
	s := config.Default()
	s.STT.Provider = "bogus"
	if _, err := New(s, 16000); err == nil {
		t.Fatal("expected STT error")
	}
	s.STT.Provider = "nemotron"
	s.TTS.Provider = "bogus"
	if _, err := New(s, 16000); err == nil {
		t.Fatal("expected TTS error")
	}
}
