package stt

import (
	"log/slog"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/dsp"
)

// ListenParams tunes the VAD-driven listening loop.
type ListenParams struct {
	Threshold       float32       // speech probability threshold
	SilenceTimeout  time.Duration // silence after speech to stop
	MinSpeech       time.Duration // minimum speech to be valid
	MaxListen       time.Duration // maximum listen duration before forcing transcription
	NoSpeechTimeout time.Duration // give up if no speech detected within this time
	SampleRate      int
	SampleWidth     int
}

// DefaultListenParams matches the Whisper provider defaults.
func DefaultListenParams(sampleRate, sampleWidth int) ListenParams {
	return ListenParams{
		Threshold:       0.4,
		SilenceTimeout:  1 * time.Second,
		MinSpeech:       300 * time.Millisecond,
		MaxListen:       60 * time.Second,
		NoSpeechTimeout: 5 * time.Second,
		SampleRate:      sampleRate,
		SampleWidth:     sampleWidth,
	}
}

// FeedResult describes what the listener decided after a chunk.
type FeedResult int

const (
	// FeedContinue: keep listening.
	FeedContinue FeedResult = iota
	// FeedEndOfSpeech: speech followed by enough silence — transcribe.
	FeedEndOfSpeech
	// FeedGiveUp: a timeout fired — stop listening and transcribe what
	// has been collected (if anything).
	FeedGiveUp
	// FeedResetSpeech: a blip too short to count — discard accumulated
	// audio and keep listening.
	FeedResetSpeech
)

// Prober computes a speech probability for a 512-sample chunk.
type Prober interface {
	Probability(st *VADState, samples []float32) (float32, error)
	NewState() *VADState
}

// VADListener runs Silero VAD over a mic stream to decide when the user
// has finished speaking.
type VADListener struct {
	vad    Prober
	params ListenParams
	state  *VADState

	vadBuf         []byte
	speechDetected bool
	speechBytes    int
	silenceStart   time.Time
	listenStart    time.Time
	onVADStart     VADStartCallback

	// Now is the clock; tests may override it.
	Now func() time.Time
}

// NewVADListener creates a listener; call Feed for each mic chunk.
func NewVADListener(vad Prober, params ListenParams, onVADStart VADStartCallback) *VADListener {
	l := &VADListener{vad: vad, params: params, state: vad.NewState(), onVADStart: onVADStart, Now: time.Now}
	l.listenStart = l.Now()
	return l
}

// SpeechDetected reports whether speech has been observed.
func (l *VADListener) SpeechDetected() bool { return l.speechDetected }

// SpeechDuration returns the accumulated speech duration.
func (l *VADListener) SpeechDuration() time.Duration {
	bytesPerSec := l.params.SampleRate * l.params.SampleWidth
	if bytesPerSec == 0 {
		return 0
	}
	return time.Duration(float64(l.speechBytes) / float64(bytesPerSec) * float64(time.Second))
}

// Feed processes one mic chunk. Timeouts are checked before the chunk is
// analysed so a FeedGiveUp result means the chunk was not consumed.
func (l *VADListener) Feed(chunk []byte) (FeedResult, error) {
	now := l.Now()
	elapsed := now.Sub(l.listenStart)
	if elapsed > l.params.MaxListen {
		slog.Warn("Max listen duration reached", "secs", l.params.MaxListen.Seconds())
		return FeedGiveUp, nil
	}
	if !l.speechDetected && elapsed > l.params.NoSpeechTimeout {
		slog.Info("No speech detected, giving up", "secs", l.params.NoSpeechTimeout.Seconds())
		return FeedGiveUp, nil
	}

	l.vadBuf = append(l.vadBuf, chunk...)
	chunkBytes := VADChunkSamples * 2
	for len(l.vadBuf) >= chunkBytes {
		vc := l.vadBuf[:chunkBytes]
		l.vadBuf = l.vadBuf[chunkBytes:]
		prob, err := l.vad.Probability(l.state, dsp.BytesToFloat32(vc))
		if err != nil {
			return FeedContinue, err
		}
		now = l.Now()
		if prob > l.params.Threshold {
			if !l.speechDetected && l.onVADStart != nil {
				l.onVADStart()
			}
			l.speechDetected = true
			l.speechBytes += chunkBytes
			l.silenceStart = time.Time{}
		} else if l.speechDetected {
			if l.silenceStart.IsZero() {
				l.silenceStart = now
			} else if now.Sub(l.silenceStart) >= l.params.SilenceTimeout {
				dur := l.SpeechDuration()
				if dur >= l.params.MinSpeech {
					slog.Info("End of speech", "speech_secs", roundSecs(dur), "silence_secs", roundSecs(now.Sub(l.silenceStart)))
					l.vadBuf = nil
					return FeedEndOfSpeech, nil
				}
				l.speechDetected = false
				l.speechBytes = 0
				l.silenceStart = time.Time{}
				l.vadBuf = nil
				return FeedResetSpeech, nil
			}
		}
	}
	return FeedContinue, nil
}

func roundSecs(d time.Duration) float64 {
	return float64(int(d.Seconds()*10)) / 10
}
