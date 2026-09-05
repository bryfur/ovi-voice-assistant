// Package audio holds the PipelineOutput interface — the sink for pipeline
// events and PCM audio destined for a device.
package audio

import (
	"context"

	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

// PipelineOutput sends pipeline events and PCM audio to a device.
type PipelineOutput interface {
	// SendEvent enqueues a control event and waits until it has been sent.
	SendEvent(ctx context.Context, event transport.EventType, payload []byte) error
	// SendAudio enqueues raw PCM for playback.
	SendAudio(ctx context.Context, pcm []byte) error
}

// DeviceOutput is a PipelineOutput bound to a single device that can be
// flushed and reset between playback sessions.
type DeviceOutput interface {
	PipelineOutput
	// Flush pads and sends any partial frame, then waits for the queue to drain.
	Flush(ctx context.Context) error
	// Reset stops the worker and discards buffered state.
	Reset()
}
