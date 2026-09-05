package pipeline

import (
	"context"
	"log/slog"
	"strings"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/audio"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/stt"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
	"github.com/bryfur/ovi-voice-assistant/internal/tts"
)

// Agent is the conversational surface the pipeline drives.
type Agent interface {
	Load() error
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	ResetHistory()
	RunStreamed(ctx context.Context, text string, actx *agent.Context, onToken func(string)) error
	RunText(ctx context.Context, text string, actx *agent.Context) (string, error)
}

// VoiceAssistant runs STT → Agent → TTS for one utterance. It works in
// PCM; codec encoding is the caller's job.
type VoiceAssistant struct {
	STT   stt.STT
	TTS   tts.TTS
	Agent Agent
}

// New creates the pipeline with the configured providers; TTS outputs at
// ttsRate.
func New(s *config.Settings, ttsRate int) (*VoiceAssistant, error) {
	st, err := stt.New(s.STT)
	if err != nil {
		return nil, err
	}
	t, err := tts.New(s.TTS, ttsRate)
	if err != nil {
		return nil, err
	}
	return &VoiceAssistant{STT: st, TTS: t, Agent: agent.New(s.LLM)}, nil
}

// Load loads STT, TTS and the agent.
func (v *VoiceAssistant) Load() error {
	for _, load := range []func() error{v.STT.Load, v.TTS.Load, v.Agent.Load} {
		if err := load(); err != nil {
			return err
		}
	}
	slog.Info("Voice pipeline ready")
	return nil
}

// Start starts MCP servers.
func (v *VoiceAssistant) Start(ctx context.Context) error { return v.Agent.Start(ctx) }

// Stop stops the agent and releases the models.
func (v *VoiceAssistant) Stop(ctx context.Context) error {
	err := v.Agent.Stop(ctx)
	v.STT.Close()
	v.TTS.Close()
	return err
}

// ResetHistory clears the conversation.
func (v *VoiceAssistant) ResetHistory() { v.Agent.ResetHistory() }

// Run handles one utterance and reports whether a follow-up was requested.
func (v *VoiceAssistant) Run(ctx context.Context, out audio.PipelineOutput, mic <-chan []byte, actx *agent.Context) bool {
	transcript, err := v.STT.Listen(ctx, mic, func() { _ = out.SendEvent(ctx, transport.EventVADStart, nil) })
	if err != nil {
		return v.fail(ctx, out, err)
	}
	_ = out.SendEvent(ctx, transport.EventMicStop, nil)
	if transcript == "" {
		slog.Info("No speech detected")
		_ = out.SendEvent(ctx, transport.EventError, []byte("stt-no-text\x00No speech detected"))
		return false
	}
	slog.Info("User said", "text", transcript)

	tokens := make(chan string, 64)
	var sb strings.Builder
	agentErr := make(chan error, 1)
	go func() {
		defer close(tokens)
		agentErr <- v.Agent.RunStreamed(ctx, transcript, actx, func(tok string) {
			sb.WriteString(tok)
			select {
			case tokens <- tok:
			case <-ctx.Done():
			}
		})
	}()

	speech := NewSpeechQueue(ctx, v.TTS, out)
	_ = out.SendEvent(ctx, transport.EventTTSStart, nil)
	err = <-speech.SubmitStream(tokens)
	speech.Stop()
	if aerr := <-agentErr; err == nil {
		err = aerr
	}
	if err != nil {
		return v.fail(ctx, out, err)
	}
	_ = out.SendEvent(ctx, transport.EventTTSEnd, nil)

	followUp := strings.Contains(sb.String(), tts.ListenToken)
	if followUp {
		_ = out.SendEvent(ctx, transport.EventContinue, nil)
	}
	slog.Info("Pipeline complete", "follow_up", followUp)
	return followUp
}

// Announce speaks text on the device.
func (v *VoiceAssistant) Announce(ctx context.Context, out audio.PipelineOutput, text string) {
	speech := NewSpeechQueue(ctx, v.TTS, out)
	_ = out.SendEvent(ctx, transport.EventTTSStart, nil)
	err := <-speech.Submit(text)
	speech.Stop()
	if err != nil {
		v.fail(ctx, out, err)
		return
	}
	_ = out.SendEvent(ctx, transport.EventTTSEnd, nil)
}

func (v *VoiceAssistant) fail(ctx context.Context, out audio.PipelineOutput, err error) bool {
	if ctx.Err() != nil {
		return false // cancelled by a new wake word or shutdown
	}
	slog.Error("Pipeline error", "err", err)
	_ = out.SendEvent(ctx, transport.EventError, []byte("pipeline_error\x00Pipeline processing failed"))
	return false
}
