package device

import (
	"context"
	"errors"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"log/slog"
	"sync"
	"time"
)

// Pacing parameters for EncodingOutput.
const (
	// leadTime is how much audio to send ahead of real-time.
	leadTime = 300 * time.Millisecond
	// queueSize bounds queued audio frames + events; pacing drives backpressure.
	queueSize = 128
)

// errOutputReset is returned to waiters when the output is reset.
var errOutputReset = errors.New("output reset")

type itemKind int

const (
	itemAudio itemKind = iota
	itemEvent
	itemBarrier
)

type outputItem struct {
	kind    itemKind
	frame   []byte
	event   EventType
	payload []byte
	done    chan error
}

// EncodingOutput adapts transport + codec to Speaker.
//
// All traffic to the device — PCM audio and control events — flows through
// a single FIFO queue drained by one worker. The worker paces encoded audio
// to real-time and inserts events in order with playback, waiting for
// real-time to catch up before emitting an event so the device doesn't cut
// off the tail of buffered audio.
type EncodingOutput struct {
	transport Transport
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
	// OnAudioConfig, if set, is called with the codec each time AUDIO_CONFIG
	// is sent to the device (before TTS_START).
	OnAudioConfig func(codec.AudioCodec)
}

// NewEncodingOutput creates an output for a transport and codec.
func NewEncodingOutput(t Transport, c codec.AudioCodec) *EncodingOutput {
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

// ensureWorker starts the worker goroutine if needed and returns the
// current queue and stop channel.
func (o *EncodingOutput) ensureWorker() (chan outputItem, chan struct{}) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.queue == nil {
		o.queue = make(chan outputItem, queueSize)
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
				slog.Debug("First audio frame to device")
			}
			frameCount++
			_ = o.transport.SendAudio(encoded)
			ahead := time.Duration(frameCount)*o.frameDuration - o.Now().Sub(t0)
			if ahead < 0 {
				t0 = o.Now().Add(-time.Duration(frameCount) * o.frameDuration)
				ahead = 0
			}
			if ahead > leadTime {
				o.Sleep(ctx, ahead-leadTime)
			}
		case itemEvent:
			catchUp()
			var err error
			if ctx.Err() != nil {
				err = errOutputReset
			} else {
				if item.event == EventTTSStart {
					if o.OnAudioConfig != nil {
						o.OnAudioConfig(o.codec)
					}
					cfg := AudioConfig{
						SampleRate:        uint32(o.codec.SampleRate()),
						EncodedFrameBytes: uint16(o.codec.EncodedFrameBytes()),
						CodecType:         o.codec.ID(),
						Channels:          uint8(o.codec.Channels()),
					}
					err = o.transport.SendEvent(EventAudioConfig, cfg.Marshal())
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
		return errOutputReset
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
		return errOutputReset
	case <-ctx.Done():
		return ctx.Err()
	}
}

// SendEvent implements Output.
func (o *EncodingOutput) SendEvent(ctx context.Context, event EventType, payload []byte) error {
	return o.enqueueAndWait(ctx, outputItem{kind: itemEvent, event: event, payload: payload})
}

// SendAudio implements Output.
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
				item.done <- errOutputReset
			}
		default:
			return
		}
	}
}
