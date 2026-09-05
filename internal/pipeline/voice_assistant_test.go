package pipeline

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/memory"
	"github.com/bryfur/ovi-voice-assistant/internal/stt"
)

type fakeSTT struct {
	transcript string
	err        error
	vadStart   bool
}

func (f *fakeSTT) Load() error                       { return nil }
func (f *fakeSTT) Transcribe([]byte) (string, error) { return f.transcript, f.err }
func (f *fakeSTT) TranscribeStream(ctx context.Context, chunks <-chan []byte, onVAD stt.VADStartCallback) (string, error) {
	if f.vadStart && onVAD != nil {
		onVAD()
	}
	return f.transcript, f.err
}

type fakeAgent struct {
	response string
	err      error
	inputs   []string
	resets   int
}

func (a *fakeAgent) Load() error                 { return nil }
func (a *fakeAgent) Start(context.Context) error { return nil }
func (a *fakeAgent) Stop(context.Context) error  { return nil }
func (a *fakeAgent) ResetHistory()               { a.resets++ }
func (a *fakeAgent) RunText(ctx context.Context, text string, actx *agent.Context) (string, error) {
	return a.response, a.err
}
func (a *fakeAgent) RunStreamed(ctx context.Context, text string, actx *agent.Context, onToken func(string)) error {
	a.inputs = append(a.inputs, text)
	if a.err != nil {
		return a.err
	}
	for _, w := range strings.SplitAfter(a.response, " ") {
		onToken(w)
	}
	return nil
}

type fakeMem struct{ retained []string }

func (m *fakeMem) Recall(context.Context, string, memory.Budget, int) (memory.RecallResult, error) {
	return memory.RecallResult{}, nil
}
func (m *fakeMem) Retain(_ context.Context, content, _ string) (memory.RetainResult, error) {
	m.retained = append(m.retained, content)
	return memory.RetainResult{Success: true}, nil
}

func newVA(stt *fakeSTT, ag *fakeAgent) *VoiceAssistant {
	return &VoiceAssistant{Settings: config.Default(), STT: stt, TTS: &slowTTS{}, Agent: ag}
}

func eventNames(out *recordingOutput) string {
	var names []string
	for _, e := range out.events {
		names = append(names, e.String())
	}
	return strings.Join(names, ",")
}

func TestRunFullPipelineSuccess(t *testing.T) {
	ag := &fakeAgent{response: "Hello there friend."}
	va := newVA(&fakeSTT{transcript: "hi", vadStart: true}, ag)
	out := &recordingOutput{}

	followUp := va.Run(context.Background(), out, nil, nil)

	if followUp {
		t.Fatal("no follow-up expected")
	}
	if got := eventNames(out); got != "VAD_START,MIC_STOP,TTS_START,TTS_END" {
		t.Fatalf("events = %s", got)
	}
	if len(out.played()) != 1 || out.played()[0] != "Hello there friend." || ag.inputs[0] != "hi" {
		t.Fatalf("played=%v inputs=%v", out.played(), ag.inputs)
	}
	if va.LastResponse() != "Hello there friend." {
		t.Fatalf("last response = %q", va.LastResponse())
	}
}

func TestRunNoSpeech(t *testing.T) {
	ag := &fakeAgent{response: "x"}
	va := newVA(&fakeSTT{transcript: ""}, ag)
	out := &recordingOutput{}

	followUp := va.Run(context.Background(), out, nil, nil)

	if followUp || len(ag.inputs) != 0 || eventNames(out) != "MIC_STOP,ERROR" {
		t.Fatalf("followUp=%v inputs=%v events=%s", followUp, ag.inputs, eventNames(out))
	}
}

func TestRunFollowUpListen(t *testing.T) {
	va := newVA(&fakeSTT{transcript: "hi"}, &fakeAgent{response: "Want more details? [LISTEN]"})
	out := &recordingOutput{}

	followUp := va.Run(context.Background(), out, nil, nil)

	if !followUp || !strings.HasSuffix(eventNames(out), "TTS_END,CONTINUE") {
		t.Fatalf("followUp=%v events=%s", followUp, eventNames(out))
	}
	if len(out.played()) != 1 || strings.Contains(out.played()[0], "[LISTEN]") {
		t.Fatalf("played = %v", out.played())
	}
}

func TestRunSTTErrorSendsPipelineError(t *testing.T) {
	va := newVA(&fakeSTT{err: errors.New("boom")}, &fakeAgent{})
	out := &recordingOutput{}

	followUp := va.Run(context.Background(), out, nil, nil)

	if followUp || eventNames(out) != "ERROR" {
		t.Fatalf("followUp=%v events=%s", followUp, eventNames(out))
	}
}

func TestRunCancelledNoErrorEvent(t *testing.T) {
	va := newVA(&fakeSTT{err: context.Canceled}, &fakeAgent{})
	out := &recordingOutput{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	followUp := va.Run(ctx, out, nil, nil)

	if followUp || len(out.events) != 0 {
		t.Fatalf("followUp=%v events=%s", followUp, eventNames(out))
	}
}

func TestRunSetsSayAndRetainsMemory(t *testing.T) {
	va := newVA(&fakeSTT{transcript: "remember this"}, &fakeAgent{response: "Noted."})
	out := &recordingOutput{}
	mem := &fakeMem{}
	actx := &agent.Context{Memory: mem}

	va.Run(context.Background(), out, nil, actx)
	va.Stop(context.Background()) // waits for background retain

	if actx.Say == nil {
		t.Fatal("Say callback not installed")
	}
	if len(mem.retained) != 1 || mem.retained[0] != "User: remember this\nAssistant: Noted." {
		t.Fatalf("retained = %v", mem.retained)
	}
}

func TestSayDuringRunIsSerialized(t *testing.T) {
	ag := &fakeAgent{response: "Final answer here."}
	va := newVA(&fakeSTT{transcript: "q"}, ag)
	out := &recordingOutput{}
	actx := &agent.Context{}
	// Simulate the agent calling say() before producing its response.
	ag2 := &sayingAgent{fakeAgent: ag, actx: actx}
	va.Agent = ag2

	va.Run(context.Background(), out, nil, actx)

	played := out.played()
	if len(played) != 2 || played[0] != "One moment." || played[1] != "Final answer here." {
		t.Fatalf("played = %v", played)
	}
}

type sayingAgent struct {
	*fakeAgent
	actx *agent.Context
}

func (s *sayingAgent) RunStreamed(ctx context.Context, text string, actx *agent.Context, onToken func(string)) error {
	_ = s.actx.Say(ctx, "One moment.")
	return s.fakeAgent.RunStreamed(ctx, text, actx, onToken)
}

func TestAnnounce(t *testing.T) {
	va := newVA(&fakeSTT{}, &fakeAgent{})
	out := &recordingOutput{}

	va.Announce(context.Background(), out, "Timer done.")

	if eventNames(out) != "TTS_START,TTS_END" || len(out.played()) != 1 || out.played()[0] != "Timer done." {
		t.Fatalf("events=%s played=%v", eventNames(out), out.played())
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
