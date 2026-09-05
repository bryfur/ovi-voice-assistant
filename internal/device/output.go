// Package audio holds the Output interface — the sink for pipeline
// events and PCM audio destined for a device.
package device

import (
	"context"
)

// Output is a sink for pipeline events and PCM audio destined for a device.
type Output interface {
	// SendEvent enqueues a control event and waits until it has been sent.
	SendEvent(ctx context.Context, event EventType, payload []byte) error
	// SendAudio enqueues raw PCM for playback.
	SendAudio(ctx context.Context, pcm []byte) error
}

// Speaker is an Output bound to one device that can be flushed and reset
// between playback sessions.
type Speaker interface {
	Output
	// Flush pads and sends any partial frame, then waits for the queue to drain.
	Flush(ctx context.Context) error
	// Reset stops the worker and discards buffered state.
	Reset()
}
