package stt

import (
	"testing"
	"time"
)

// fakeProber reports speech for chunks whose first sample is non-zero.
type fakeProber struct{}

func (fakeProber) NewState() *VADState { return &VADState{} }
func (fakeProber) Probability(_ *VADState, samples []float32) (float32, error) {
	if samples[0] != 0 {
		return 0.9, nil
	}
	return 0.1, nil
}

func speech(n int) []byte {
	b := make([]byte, n)
	for i := 0; i < n; i += 2 {
		b[i] = 0x10
		b[i+1] = 0x10
	}
	return b
}

func silence(n int) []byte { return make([]byte, n) }

func newListener(t *testing.T) (*VADListener, *time.Time) {
	t.Helper()
	now := time.Unix(1000, 0)
	l := NewVADListener(fakeProber{}, DefaultListenParams(16000, 2), nil)
	l.Now = func() time.Time { return now }
	l.listenStart = now
	return l, &now
}

func TestEndOfSpeechAfterSilence(t *testing.T) {
	l, now := newListener(t)
	started := false
	l.onVADStart = func() { started = true }
	chunk := VADChunkSamples * 2

	for i := 0; i < 20; i++ { // 0.64 s of speech
		if r, _ := l.Feed(speech(chunk)); r != FeedContinue {
			t.Fatalf("unexpected %v during speech", r)
		}
	}
	r1, _ := l.Feed(silence(chunk))
	*now = now.Add(1100 * time.Millisecond)
	r2, _ := l.Feed(silence(chunk))

	if !started || r1 != FeedContinue || r2 != FeedEndOfSpeech {
		t.Fatalf("started=%v r1=%v r2=%v", started, r1, r2)
	}
}

func TestShortBlipResetsSpeech(t *testing.T) {
	l, now := newListener(t)
	chunk := VADChunkSamples * 2

	l.Feed(speech(chunk)) // 32 ms < MinSpeech
	l.Feed(silence(chunk))
	*now = now.Add(1100 * time.Millisecond)
	r, _ := l.Feed(silence(chunk))

	if r != FeedResetSpeech || l.SpeechDetected() {
		t.Fatalf("r=%v detected=%v", r, l.SpeechDetected())
	}
}

func TestNoSpeechTimeoutGivesUp(t *testing.T) {
	l, now := newListener(t)

	*now = now.Add(6 * time.Second)
	r, _ := l.Feed(silence(VADChunkSamples * 2))

	if r != FeedGiveUp {
		t.Fatalf("got %v", r)
	}
}

func TestMaxListenGivesUp(t *testing.T) {
	l, now := newListener(t)
	l.Feed(speech(VADChunkSamples * 2))

	*now = now.Add(61 * time.Second)
	r, _ := l.Feed(speech(VADChunkSamples * 2))

	if r != FeedGiveUp {
		t.Fatalf("got %v", r)
	}
}

func TestPartialChunksAccumulate(t *testing.T) {
	l, _ := newListener(t)

	r, err := l.Feed(speech(100))

	if err != nil || r != FeedContinue || l.SpeechDetected() {
		t.Fatalf("partial chunk should not be analysed yet: r=%v detected=%v", r, l.SpeechDetected())
	}
	l.Feed(speech(VADChunkSamples*2 - 100))
	if !l.SpeechDetected() {
		t.Fatal("speech should be detected once a full chunk is available")
	}
}
