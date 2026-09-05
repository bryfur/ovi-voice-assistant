package pipeline

import (
	"context"
	"errors"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/speech"
	"log/slog"
	"sync"
)

// errSpeechQueueStopped is returned for submissions after Stop.
var errSpeechQueueStopped = errors.New("speech queue stopped")

type speechItem struct {
	tokens <-chan string
	done   chan error
}

// speechQueue is a single-worker queue that serializes TTS synthesis and
// device playback.
//
// Multiple callers (say tool, response TTS, announcements) submit text and
// return immediately. One worker processes submissions in FIFO order,
// streaming audio to the output so utterances never overlap on the wire.
type speechQueue struct {
	ctx    context.Context
	tts    speech.TTS
	output device.Output

	// mu is held for reading while a submission is in flight and for
	// writing by Stop, so the queue channel is never closed under a sender.
	mu     sync.RWMutex
	queue  chan speechItem
	done   chan struct{}
	closed bool
}

// newSpeechQueue creates a queue bound to ctx; Stop must be called.
func newSpeechQueue(ctx context.Context, t speech.TTS, output device.Output) *speechQueue {
	return &speechQueue{ctx: ctx, tts: t, output: output}
}

// start launches the worker on first use. Caller must hold mu (read lock
// suffices; the double-checked upgrade below handles creation).
func (q *speechQueue) start() {
	q.mu.RUnlock()
	q.mu.Lock()
	if q.queue == nil && !q.closed {
		q.queue = make(chan speechItem, 64)
		q.done = make(chan struct{})
		go q.run(q.queue, q.done)
	}
	q.mu.Unlock()
	q.mu.RLock()
}

// Submit enqueues a fixed utterance. The returned channel receives the
// playback result once the utterance has been sent.
func (q *speechQueue) Submit(text string) <-chan error {
	tokens := make(chan string, 1)
	tokens <- text
	close(tokens)
	return q.SubmitStream(tokens)
}

// SubmitStream enqueues a token stream for synthesis.
func (q *speechQueue) SubmitStream(tokens <-chan string) <-chan error {
	done := make(chan error, 1)
	q.mu.RLock()
	defer q.mu.RUnlock()
	if q.queue == nil {
		q.start()
	}
	if q.closed {
		done <- errSpeechQueueStopped
		return done
	}
	select {
	case q.queue <- speechItem{tokens: tokens, done: done}:
	case <-q.ctx.Done():
		done <- q.ctx.Err()
	}
	return done
}

// Stop closes the queue and waits for queued utterances to finish playing.
func (q *speechQueue) Stop() {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return
	}
	q.closed = true
	queue := q.queue
	done := q.done
	if queue != nil {
		close(queue)
	}
	q.mu.Unlock()
	if done != nil {
		<-done
	}
}

func (q *speechQueue) run(queue chan speechItem, done chan struct{}) {
	defer close(done)
	for item := range queue {
		err := speech.Stream(q.ctx, q.tts, item.tokens, func(pcm []byte) error {
			return q.output.SendAudio(q.ctx, pcm)
		})
		if err != nil && q.ctx.Err() == nil {
			slog.Error("speechQueue worker error", "err", err)
		}
		item.done <- err
	}
}
