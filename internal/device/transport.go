package device

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// Handler receives what a device sends. Callbacks run on the transport's
// receive goroutine and must return promptly.
type Handler struct {
	Event      func(e Event, payload []byte)
	Audio      func(frame []byte)
	Connect    func() // called after an automatic reconnect
	Disconnect func()
}

// Transport carries events and audio frames to and from one device.
type Transport interface {
	fmt.Stringer
	// Connect opens the link and starts delivering to h, reconnecting on
	// its own if the link drops.
	Connect(h Handler) error
	// Disconnect closes the link and stops reconnecting.
	Disconnect() error
	SendEvent(e Event, payload []byte) error
	SendAudio(frame []byte) error
}

// reconnect retries dial after every delay until it succeeds or ctx ends,
// then reports the new link through h.Connect.
func reconnect(ctx context.Context, what string, delay time.Duration, dial func() error, h Handler) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(delay):
		}
		slog.Info("Reconnecting", "to", what)
		if err := dial(); err != nil {
			slog.Warn("Reconnect failed, retrying", "to", what, "in", delay, "err", err)
			continue
		}
		slog.Info("Reconnected", "to", what)
		if h.Connect != nil {
			h.Connect()
		}
		return
	}
}
