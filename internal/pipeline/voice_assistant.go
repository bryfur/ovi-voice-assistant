// Package pipeline runs the voice loop for each connected device: one
// Connection per device turns wake words into sessions, a VoiceAssistant
// runs STT, the agent and TTS for each utterance, and a Manager lets only
// the device that heard the wake word best answer.
package pipeline

import (
	"context"
	"log/slog"
	"strings"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/stt"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/tts"
)

// Agent is the conversational surface the pipeline drives.
type Agent interface {
	Load() error
	Start(ctx context.Context) error
	Stop()
	Reset()
	Run(ctx context.Context, text string, env *agent.Env, emit func(string)) error
	Ask(ctx context.Context, text string, env *agent.Env) (string, error)
}

// VoiceAssistant answers one utterance at a time: listen, think, speak.
// It works in PCM; encoding for the device is the Output's job.
type VoiceAssistant struct {
	STT   stt.Recognizer
	TTS   tts.Synthesizer
	Agent Agent
}

// New builds the configured providers. TTS renders at the rate the device
// codec will use (LC3 snaps 22050 Hz to 24 kHz).
func New(s *config.Settings) (*VoiceAssistant, error) {
	probe, err := codec.New(s.Transport.Codec, 24000, 1, 0)
	if err != nil {
		return nil, err
	}
	recognizer, err := stt.New(s.STT)
	if err != nil {
		return nil, err
	}
	synth, err := tts.New(s.TTS, probe.Format().Rate)
	if err != nil {
		return nil, err
	}
	return &VoiceAssistant{STT: recognizer, TTS: synth, Agent: agent.New(s.LLM)}, nil
}

// Rate is the sample rate of the speech produced.
func (v *VoiceAssistant) Rate() int { return v.TTS.SampleRate() }

// Load loads the models and the agent configuration.
func (v *VoiceAssistant) Load() error {
	for _, load := range []func() error{v.STT.Load, v.TTS.Load, v.Agent.Load} {
		if err := load(); err != nil {
			return err
		}
	}
	slog.Info("Voice pipeline ready")
	return nil
}

// Start launches the agent's MCP servers.
func (v *VoiceAssistant) Start(ctx context.Context) error { return v.Agent.Start(ctx) }

// Stop releases the models and stops the agent.
func (v *VoiceAssistant) Stop() {
	v.Agent.Stop()
	v.STT.Close()
	v.TTS.Close()
}

// Reset starts a fresh conversation.
func (v *VoiceAssistant) Reset() { v.Agent.Reset() }

// Run handles one utterance from mic and reports whether the reply asked
// for a follow-up.
func (v *VoiceAssistant) Run(ctx context.Context, out device.Output, mic <-chan []byte, env *agent.Env) bool {
	text, err := v.STT.Listen(ctx, mic, func() { _ = out.SendEvent(ctx, device.EventVADStart, nil) })
	if err != nil {
		return v.fail(ctx, out, err)
	}
	_ = out.SendEvent(ctx, device.EventMicStop, nil)
	if text == "" {
		slog.Info("No speech detected")
		_ = out.SendEvent(ctx, device.EventError, []byte("stt-no-text\x00No speech detected"))
		return false
	}
	slog.Info("User said", "text", text)

	tokens := make(chan string, 64)
	var reply strings.Builder
	thought := make(chan error, 1)
	go func() {
		defer close(tokens)
		thought <- v.Agent.Run(ctx, text, env, func(tok string) {
			reply.WriteString(tok)
			select {
			case tokens <- tok:
			case <-ctx.Done():
			}
		})
	}()
	_ = out.SendEvent(ctx, device.EventTTSStart, nil)
	err = v.speak(ctx, out, tokens)
	if err != nil {
		go func() { // let the agent finish talking to nobody
			for range tokens {
			}
		}()
	}
	if agentErr := <-thought; err == nil {
		err = agentErr
	}
	if err != nil {
		return v.fail(ctx, out, err)
	}
	_ = out.SendEvent(ctx, device.EventTTSEnd, nil)
	again := strings.Contains(reply.String(), tts.ListenToken)
	if again {
		_ = out.SendEvent(ctx, device.EventContinue, nil)
	}
	slog.Info("Pipeline complete", "follow_up", again)
	return again
}

// Announce speaks text on the device.
func (v *VoiceAssistant) Announce(ctx context.Context, out device.Output, text string) {
	tokens := make(chan string, 1)
	tokens <- text
	close(tokens)
	_ = out.SendEvent(ctx, device.EventTTSStart, nil)
	if err := v.speak(ctx, out, tokens); err != nil {
		v.fail(ctx, out, err)
		return
	}
	_ = out.SendEvent(ctx, device.EventTTSEnd, nil)
}

func (v *VoiceAssistant) speak(ctx context.Context, out device.Output, tokens <-chan string) error {
	return tts.Stream(ctx, v.TTS, tokens, func(pcm []byte) error { return out.SendAudio(ctx, pcm) })
}

// fail reports an error to the device, unless the session was cancelled.
func (v *VoiceAssistant) fail(ctx context.Context, out device.Output, err error) bool {
	if ctx.Err() == nil {
		slog.Error("Pipeline error", "err", err)
		_ = out.SendEvent(ctx, device.EventError, []byte("pipeline_error\x00Pipeline processing failed"))
	}
	return false
}
