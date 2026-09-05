// Package pipeline runs the voice pipeline: STT → Agent → TTS with
// event-driven, real-time paced output to a device.
package pipeline

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

// Pacing parameters for EncodingOutput.
const (
	// LeadTime is how much audio to send ahead of real-time.
	LeadTime = 300 * time.Millisecond
	// QueueSize bounds queued audio frames + events; pacing drives backpressure.
	QueueSize = 128
)

// ErrOutputReset is returned to waiters when the output is reset.
var ErrOutputReset = errors.New("output reset")

type itemKind int

const (
	itemAudio itemKind = iota
	itemEvent
	itemBarrier
)

type outputItem struct {
	kind    itemKind
	frame   []byte
	event   transport.EventType
	payload []byte
	done    chan error
}

// EncodingOutput adapts transport + codec to audio.DeviceOutput.
//
// All traffic to the device — PCM audio and control events — flows through
// a single FIFO queue drained by one worker. The worker paces encoded audio
// to real-time and inserts events in order with playback, waiting for
// real-time to catch up before emitting an event so the device doesn't cut
// off the tail of buffered audio.
type EncodingOutput struct {
	transport transport.DeviceTransport
	codec     codec.AudioCodec

	mu            sync.Mutex
	pcmBuf        []byte
	queue         chan outputItem
	stop          chan struct{}
	workerDone    chan struct{}
	frameDuration time.Duration

	// Sleep is the pacing sleep; tests may replace it.
	Sleep func(ctx context.Context, d time.Duration)
	// Now is the clock; tests may replace it.
	Now func() time.Time
}

// NewEncodingOutput creates an output for a transport and codec.
func NewEncodingOutput(t transport.DeviceTransport, c codec.AudioCodec) *EncodingOutput {
	o := &EncodingOutput{
		transport:     t,
		codec:         c,
		frameDuration: time.Duration(c.FrameDurationMs()) * time.Millisecond,
		Now:           time.Now,
	}
	o.Sleep = func(ctx context.Context, d time.Duration) {
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-t.C:
		case <-ctx.Done():
		}
	}
	return o
}

// Codec returns the output codec.
func (o *EncodingOutput) Codec() codec.AudioCodec { return o.codec }

// ensureWorker starts the worker goroutine if needed and returns the
// current queue and stop channel.
func (o *EncodingOutput) ensureWorker() (chan outputItem, chan struct{}) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.queue == nil {
		o.queue = make(chan outputItem, QueueSize)
		o.stop = make(chan struct{})
		o.workerDone = make(chan struct{})
		go o.run(o.queue, o.stop, o.workerDone)
	}
	return o.queue, o.stop
}

func (o *EncodingOutput) run(queue chan outputItem, stop chan struct{}, done chan struct{}) {
	defer close(done)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-stop
		cancel()
	}()

	var frameCount int
	var t0 time.Time
	catchUp := func() {
		if frameCount == 0 {
			return
		}
		ahead := time.Duration(frameCount)*o.frameDuration - o.Now().Sub(t0)
		if ahead > 0 {
			o.Sleep(ctx, ahead)
		}
	}

	for {
		var item outputItem
		select {
		case <-stop:
			return
		case item = <-queue:
		}
		switch item.kind {
		case itemAudio:
			encoded, err := o.codec.Encode(item.frame)
			if err != nil {
				continue
			}
			if frameCount == 0 {
				t0 = o.Now()
			}
			frameCount++
			_ = o.transport.SendAudio(encoded)
			ahead := time.Duration(frameCount)*o.frameDuration - o.Now().Sub(t0)
			if ahead < 0 {
				t0 = o.Now().Add(-time.Duration(frameCount) * o.frameDuration)
				ahead = 0
			}
			if ahead > LeadTime {
				o.Sleep(ctx, ahead-LeadTime)
			}
		case itemEvent:
			catchUp()
			var err error
			if ctx.Err() != nil {
				err = ErrOutputReset
			} else {
				if item.event == transport.EventTTSStart {
					cfg := transport.AudioConfig{
						SampleRate:        uint32(o.codec.SampleRate()),
						EncodedFrameBytes: uint16(o.codec.EncodedFrameBytes()),
						CodecType:         o.codec.ID(),
						Channels:          uint8(o.codec.Channels()),
					}
					err = o.transport.SendEvent(transport.EventAudioConfig, cfg.Marshal())
				}
				if err == nil {
					err = o.transport.SendEvent(item.event, item.payload)
				}
			}
			item.done <- err
		case itemBarrier:
			item.done <- nil
		}
	}
}

func (o *EncodingOutput) enqueue(ctx context.Context, item outputItem) error {
	queue, stop := o.ensureWorker()
	select {
	case queue <- item:
		return nil
	case <-stop:
		return ErrOutputReset
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (o *EncodingOutput) enqueueAndWait(ctx context.Context, item outputItem) error {
	item.done = make(chan error, 1)
	if err := o.enqueue(ctx, item); err != nil {
		return err
	}
	_, stop := o.ensureWorker()
	select {
	case err := <-item.done:
		return err
	case <-stop:
		return ErrOutputReset
	case <-ctx.Done():
		return ctx.Err()
	}
}

// SendEvent implements audio.PipelineOutput.
func (o *EncodingOutput) SendEvent(ctx context.Context, event transport.EventType, payload []byte) error {
	return o.enqueueAndWait(ctx, outputItem{kind: itemEvent, event: event, payload: payload})
}

// SendAudio implements audio.PipelineOutput.
func (o *EncodingOutput) SendAudio(ctx context.Context, pcm []byte) error {
	frameBytes := o.codec.PCMFrameBytes()
	o.mu.Lock()
	o.pcmBuf = append(o.pcmBuf, pcm...)
	var frames [][]byte
	for len(o.pcmBuf) >= frameBytes {
		frame := make([]byte, frameBytes)
		copy(frame, o.pcmBuf[:frameBytes])
		o.pcmBuf = o.pcmBuf[frameBytes:]
		frames = append(frames, frame)
	}
	o.mu.Unlock()
	for _, frame := range frames {
		if err := o.enqueue(ctx, outputItem{kind: itemAudio, frame: frame}); err != nil {
			return err
		}
	}
	return nil
}

// Flush pads and sends any partial frame, then waits for the queue to drain.
func (o *EncodingOutput) Flush(ctx context.Context) error {
	frameBytes := o.codec.PCMFrameBytes()
	o.mu.Lock()
	var tail []byte
	if len(o.pcmBuf) > 0 {
		tail = make([]byte, frameBytes)
		copy(tail, o.pcmBuf)
		o.pcmBuf = nil
	}
	running := o.queue != nil
	o.mu.Unlock()
	if tail != nil {
		if err := o.enqueue(ctx, outputItem{kind: itemAudio, frame: tail}); err != nil {
			return err
		}
		running = true
	}
	if !running {
		return nil
	}
	return o.enqueueAndWait(ctx, outputItem{kind: itemBarrier})
}

// Reset stops the worker, discards buffered audio and queued items, and
// zeroes the pacing state.
func (o *EncodingOutput) Reset() {
	o.mu.Lock()
	stop := o.stop
	done := o.workerDone
	queue := o.queue
	o.queue = nil
	o.stop = nil
	o.workerDone = nil
	o.pcmBuf = nil
	o.mu.Unlock()
	if stop == nil {
		return
	}
	close(stop)
	<-done
	// Unblock any waiters on discarded items.
	for {
		select {
		case item := <-queue:
			if item.done != nil {
				item.done <- ErrOutputReset
			}
		default:
			return
		}
	}
}
