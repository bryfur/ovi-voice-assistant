package device

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
)

// Output is where the pipeline sends what a device should play: control
// events and raw PCM, delivered in order.
type Output interface {
	// SendEvent queues an event behind any audio already queued and waits
	// until it has been sent.
	SendEvent(ctx context.Context, e Event, payload []byte) error
	// SendAudio queues PCM for playback.
	SendAudio(ctx context.Context, pcm []byte) error
	// Flush pads the last partial frame and waits for the queue to drain.
	Flush(ctx context.Context) error
	// Reset drops everything queued and forgets the pacing clock.
	Reset()
}

const (
	lead      = 300 * time.Millisecond // how far ahead of real time audio is sent
	queueSize = 128                    // frames + events in flight; pacing provides backpressure
)

var errReset = errors.New("speaker reset")

// Speaker encodes PCM for one device and paces it to real time. Audio and
// events share one queue drained by one goroutine, so an event is sent
// only once playback has caught up with the audio queued before it and
// the device never cuts off a tail of buffered speech.
type Speaker struct {
	t     Transport
	codec codec.Codec
	f     codec.Format
	// OnConfig, if set, sees the format each time it is announced to the device.
	OnConfig func(codec.Format)

	mu     sync.Mutex
	pcm    []byte // partial frame
	jobs   chan func(context.Context)
	ctx    context.Context // ends when the queue is reset
	cancel context.CancelFunc
	done   chan struct{}

	// Pacing state, touched only by the queue goroutine.
	frames int
	start  time.Time
	now    func() time.Time
	sleep  func(context.Context, time.Duration)
}

// NewSpeaker pairs a transport with the codec its device decodes.
func NewSpeaker(t Transport, c codec.Codec) *Speaker {
	return &Speaker{t: t, codec: c, f: c.Format(), now: time.Now, sleep: sleep}
}

func sleep(ctx context.Context, d time.Duration) {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
}

// Format is the audio format sent to the device.
func (s *Speaker) Format() codec.Format { return s.f }

func (s *Speaker) SendAudio(ctx context.Context, pcm []byte) error {
	size := s.f.PCMBytes()
	s.mu.Lock()
	s.pcm = append(s.pcm, pcm...)
	var frames [][]byte
	for len(s.pcm) >= size {
		frames = append(frames, append([]byte(nil), s.pcm[:size]...))
		s.pcm = s.pcm[size:]
	}
	s.mu.Unlock()
	for _, frame := range frames {
		if err := s.submit(ctx, func(ctx context.Context) { s.play(ctx, frame) }); err != nil {
			return err
		}
	}
	return nil
}

func (s *Speaker) SendEvent(ctx context.Context, e Event, payload []byte) error {
	return s.await(ctx, func(ctx context.Context) error {
		s.catchUp(ctx)
		if ctx.Err() != nil {
			return errReset
		}
		if e == EventTTSStart {
			if s.OnConfig != nil {
				s.OnConfig(s.f)
			}
			if err := s.t.SendEvent(EventAudioConfig, AudioConfig(s.f)); err != nil {
				return err
			}
		}
		return s.t.SendEvent(e, payload)
	})
}

func (s *Speaker) Flush(ctx context.Context) error {
	s.mu.Lock()
	tail, running := s.pcm, s.jobs != nil
	s.pcm = nil
	s.mu.Unlock()
	if len(tail) > 0 {
		frame := make([]byte, s.f.PCMBytes())
		copy(frame, tail)
		if err := s.submit(ctx, func(ctx context.Context) { s.play(ctx, frame) }); err != nil {
			return err
		}
		running = true
	}
	if !running {
		return nil
	}
	return s.await(ctx, func(context.Context) error { return nil })
}

func (s *Speaker) Reset() {
	s.mu.Lock()
	cancel, done := s.cancel, s.done
	s.jobs, s.ctx, s.cancel, s.done, s.pcm = nil, nil, nil, nil, nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
}

// queue returns the job queue and its context, starting the goroutine on
// first use.
func (s *Speaker) queue() (chan func(context.Context), context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.jobs == nil {
		s.jobs = make(chan func(context.Context), queueSize)
		s.ctx, s.cancel = context.WithCancel(context.Background())
		s.done = make(chan struct{})
		go s.run(s.jobs, s.ctx, s.done)
	}
	return s.jobs, s.ctx
}

func (s *Speaker) run(jobs chan func(context.Context), ctx context.Context, done chan struct{}) {
	defer close(done)
	s.frames = 0
	for {
		select {
		case <-ctx.Done():
			return
		case job := <-jobs:
			job(ctx)
		}
	}
}

func (s *Speaker) submit(ctx context.Context, job func(context.Context)) error {
	jobs, queue := s.queue()
	select {
	case jobs <- job:
		return nil
	case <-queue.Done():
		return errReset
	case <-ctx.Done():
		return ctx.Err()
	}
}

// await runs job on the queue and returns its result.
func (s *Speaker) await(ctx context.Context, job func(context.Context) error) error {
	result := make(chan error, 1)
	_, queue := s.queue()
	if err := s.submit(ctx, func(ctx context.Context) { result <- job(ctx) }); err != nil {
		return err
	}
	select {
	case err := <-result:
		return err
	case <-queue.Done():
		return errReset
	case <-ctx.Done():
		return ctx.Err()
	}
}

// play encodes and sends one frame, then sleeps so that transmission stays
// at most lead ahead of playback.
func (s *Speaker) play(ctx context.Context, frame []byte) {
	encoded, err := s.codec.Encode(frame)
	if err != nil {
		slog.Warn("Encode failed", "err", err)
		return
	}
	if s.frames == 0 {
		s.start = s.now()
		slog.Debug("First audio frame to device")
	}
	s.frames++
	_ = s.t.SendAudio(encoded)
	if ahead := s.ahead(); ahead < 0 {
		s.start = s.now().Add(-s.played()) // fell behind: restart the clock
	} else if ahead > lead {
		s.sleep(ctx, ahead-lead)
	}
}

// catchUp waits until everything sent so far has played out.
func (s *Speaker) catchUp(ctx context.Context) {
	if ahead := s.ahead(); s.frames > 0 && ahead > 0 {
		s.sleep(ctx, ahead)
	}
}

func (s *Speaker) played() time.Duration {
	return time.Duration(s.frames) * time.Duration(s.f.FrameMs) * time.Millisecond
}
func (s *Speaker) ahead() time.Duration { return s.played() - s.now().Sub(s.start) }
