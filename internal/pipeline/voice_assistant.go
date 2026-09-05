package pipeline

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/audio"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/stt"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
	"github.com/bryfur/ovi-voice-assistant/internal/tts"
)

// ListenToken marks a response that expects a follow-up from the user.
const ListenToken = "[LISTEN]"

// Agent is the conversational agent surface the pipeline needs.
type Agent interface {
	Load() error
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	ResetHistory()
	RunStreamed(ctx context.Context, text string, actx *agent.Context, onToken func(string)) error
	RunText(ctx context.Context, text string, actx *agent.Context) (string, error)
}

// VoiceAssistant loads models and runs the full voice pipeline
// (STT → Agent → TTS). It works entirely in PCM — codec encoding/decoding
// is handled by the caller.
type VoiceAssistant struct {
	Settings *config.Settings
	STT      stt.STT
	TTS      tts.TTS
	Agent    Agent

	mu           sync.Mutex
	lastResponse string
	background   sync.WaitGroup
}

// New creates the pipeline with the configured providers. TTS is created
// at ttsSampleRate.
func New(settings *config.Settings, ttsSampleRate int) (*VoiceAssistant, error) {
	s, err := stt.Create(settings)
	if err != nil {
		return nil, err
	}
	t, err := tts.Create(settings, ttsSampleRate)
	if err != nil {
		return nil, err
	}
	return &VoiceAssistant{
		Settings: settings,
		STT:      s,
		TTS:      t,
		Agent:    agent.New(settings),
	}, nil
}

// Load loads STT, TTS and agent models.
func (v *VoiceAssistant) Load() error {
	if err := v.STT.Load(); err != nil {
		return err
	}
	if err := v.TTS.Load(); err != nil {
		return err
	}
	if err := v.Agent.Load(); err != nil {
		return err
	}
	slog.Info("Voice pipeline ready")
	return nil
}

// Start starts the agent (MCP servers).
func (v *VoiceAssistant) Start(ctx context.Context) error { return v.Agent.Start(ctx) }

// Stop stops the agent and waits for background work.
func (v *VoiceAssistant) Stop(ctx context.Context) error {
	err := v.Agent.Stop(ctx)
	done := make(chan struct{})
	go func() {
		v.background.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
	}
	return err
}

// ResetHistory clears the agent's conversation history.
func (v *VoiceAssistant) ResetHistory() { v.Agent.ResetHistory() }

// LastResponse returns the last spoken agent response.
func (v *VoiceAssistant) LastResponse() string {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.lastResponse
}

// Run executes the full pipeline for one utterance. It returns true if a
// follow-up listen was requested.
func (v *VoiceAssistant) Run(ctx context.Context, output audio.PipelineOutput, mic <-chan []byte, actx *agent.Context) bool {
	speech := NewSpeechQueue(ctx, v.TTS, output)
	defer speech.Stop()

	transcript, err := v.runSTT(ctx, output, mic)
	if err != nil {
		return v.fail(ctx, output, err)
	}
	if transcript == "" {
		return false
	}

	if actx != nil {
		actx.Say = func(_ context.Context, text string) error {
			speech.Submit(text)
			return nil
		}
	}

	tokens := make(chan string, 64)
	agentErr := make(chan error, 1)
	go func() {
		defer close(tokens)
		agentErr <- v.Agent.RunStreamed(ctx, transcript, actx, func(tok string) {
			select {
			case tokens <- tok:
			case <-ctx.Done():
			}
		})
	}()

	followUp, fullText, err := v.streamTTS(ctx, output, speech, tokens)
	if aerr := <-agentErr; err == nil && aerr != nil {
		err = aerr
	}
	if err != nil {
		return v.fail(ctx, output, err)
	}
	v.mu.Lock()
	v.lastResponse = fullText
	v.mu.Unlock()

	if followUp {
		if err := output.SendEvent(ctx, transport.EventContinue, nil); err != nil {
			return v.fail(ctx, output, err)
		}
	}

	// Auto-retain: fire-and-forget memory extraction
	if actx != nil && actx.Memory != nil {
		exchange := "User: " + transcript + "\nAssistant: " + fullText
		mem := actx.Memory
		v.background.Add(1)
		go func() {
			defer v.background.Done()
			if _, err := mem.Retain(context.Background(), exchange, "voice conversation"); err != nil {
				slog.Error("Background memory retain failed", "err", err)
			}
		}()
	}

	slog.Info("Pipeline complete", "follow_up", followUp)
	return followUp
}

// fail logs a pipeline error and notifies the device unless cancelled.
func (v *VoiceAssistant) fail(ctx context.Context, output audio.PipelineOutput, err error) bool {
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		slog.Debug("Pipeline cancelled")
		return false
	}
	slog.Error("Pipeline error", "err", err)
	_ = output.SendEvent(ctx, transport.EventError, []byte("pipeline_error\x00Pipeline processing failed"))
	return false
}

// Announce plays a TTS announcement.
func (v *VoiceAssistant) Announce(ctx context.Context, output audio.PipelineOutput, text string) {
	speech := NewSpeechQueue(ctx, v.TTS, output)
	defer speech.Stop()
	if err := output.SendEvent(ctx, transport.EventTTSStart, nil); err != nil {
		v.announceErr(ctx, err)
		return
	}
	if err := <-speech.Submit(text); err != nil {
		v.announceErr(ctx, err)
		return
	}
	if err := output.SendEvent(ctx, transport.EventTTSEnd, nil); err != nil {
		v.announceErr(ctx, err)
		return
	}
	slog.Info("Announcement complete", "text", truncate(text, 60))
}

func (v *VoiceAssistant) announceErr(ctx context.Context, err error) {
	if ctx.Err() != nil {
		slog.Debug("Announcement cancelled")
		return
	}
	slog.Error("Announcement error", "err", err)
}

// -- Internal --

func (v *VoiceAssistant) runSTT(ctx context.Context, output audio.PipelineOutput, mic <-chan []byte) (string, error) {
	transcript, err := v.STT.TranscribeStream(ctx, mic, func() {
		_ = output.SendEvent(ctx, transport.EventVADStart, nil)
	})
	if err != nil {
		return "", err
	}
	if err := output.SendEvent(ctx, transport.EventMicStop, nil); err != nil {
		return "", err
	}
	if transcript == "" {
		slog.Info("No speech detected")
		if err := output.SendEvent(ctx, transport.EventError, []byte("stt-no-text\x00No speech detected")); err != nil {
			return "", err
		}
		return "", nil
	}
	slog.Debug("User said", "text", transcript)
	return transcript, nil
}

// streamTTS forwards agent tokens to the speech queue, returning whether a
// follow-up was requested and the full response text.
func (v *VoiceAssistant) streamTTS(ctx context.Context, output audio.PipelineOutput, speech *SpeechQueue, tokens <-chan string) (bool, string, error) {
	filtered := make(chan string, 64)
	var sb strings.Builder
	var mu sync.Mutex
	firstToken := make(chan struct{})
	go func() {
		defer close(filtered)
		first := true
		for tok := range tokens {
			mu.Lock()
			sb.WriteString(tok)
			mu.Unlock()
			if first {
				close(firstToken)
				first = false
			}
			select {
			case filtered <- tok:
			case <-ctx.Done():
				for range tokens {
				}
				return
			}
		}
		if first {
			close(firstToken)
		}
	}()

	if err := output.SendEvent(ctx, transport.EventTTSStart, nil); err != nil {
		return false, "", err
	}
	// Submit the response stream only once the agent has started talking, so
	// a say() issued before the answer (e.g. ahead of a slow tool call) is
	// queued — and spoken — first.
	select {
	case <-firstToken:
	case <-ctx.Done():
		return false, "", ctx.Err()
	}
	if err := <-speech.SubmitStream(filtered); err != nil {
		return false, "", err
	}
	if err := output.SendEvent(ctx, transport.EventTTSEnd, nil); err != nil {
		return false, "", err
	}
	mu.Lock()
	fullText := sb.String()
	mu.Unlock()
	slog.Info("Speaking complete", "text", truncate(fullText, 80))
	return strings.Contains(fullText, ListenToken), fullText, nil
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
