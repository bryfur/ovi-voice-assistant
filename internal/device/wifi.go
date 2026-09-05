package device

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"strconv"
	"sync"
	"time"
)

// WiFi speaks the length-prefixed TCP protocol: the device listens on port
// 6055 and Ovi connects as the client. Every frame is [2B LE length][type
// byte][payload]; the type byte is an Event or one of the audio markers.
type WiFi struct {
	addr  string
	retry time.Duration

	mu      sync.Mutex
	h       Handler
	conn    net.Conn      // nil while disconnected
	done    chan struct{} // closed when the current receive loop ends
	closing bool
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewWiFi addresses a device by host and port (0 = 6055).
func NewWiFi(host string, port int) *WiFi {
	return &WiFi{addr: net.JoinHostPort(host, strconv.Itoa(max(port, 6055))), retry: 3 * time.Second}
}

func (t *WiFi) String() string { return t.addr }

func (t *WiFi) Connect(h Handler) error {
	t.mu.Lock()
	t.h, t.closing = h, false
	t.ctx, t.cancel = context.WithCancel(context.Background())
	t.mu.Unlock()
	return t.dial()
}

func (t *WiFi) Disconnect() error {
	t.mu.Lock()
	t.closing = true
	t.cancel()
	conn, done := t.conn, t.done
	t.conn = nil
	t.mu.Unlock()
	if conn != nil {
		_ = conn.Close()
		<-done
	}
	return nil
}

func (t *WiFi) SendEvent(e Event, payload []byte) error { return t.send(byte(e), payload) }
func (t *WiFi) SendAudio(frame []byte) error            { return t.send(speakerAudio, frame) }

func (t *WiFi) send(kind byte, payload []byte) error {
	if len(payload) > 0xFFFE {
		return fmt.Errorf("frame too large: %d bytes", len(payload))
	}
	buf := make([]byte, 3+len(payload))
	binary.LittleEndian.PutUint16(buf, uint16(1+len(payload)))
	buf[2] = kind
	copy(buf[3:], payload)
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn == nil {
		slog.Warn("Not connected, dropping frame", "to", t.addr, "type", fmt.Sprintf("0x%02X", kind))
		return nil
	}
	_, err := t.conn.Write(buf)
	return err
}

// dial opens the TCP link with keepalive so a dead device is noticed in
// ~25 s, then starts the receive loop.
func (t *WiFi) dial() error {
	d := net.Dialer{Timeout: 10 * time.Second, KeepAliveConfig: net.KeepAliveConfig{
		Enable: true, Idle: 10 * time.Second, Interval: 5 * time.Second, Count: 3,
	}}
	conn, err := d.Dial("tcp", t.addr)
	if err != nil {
		return err
	}
	t.mu.Lock()
	if t.closing {
		t.mu.Unlock()
		return conn.Close()
	}
	t.conn, t.done = conn, make(chan struct{})
	done := t.done
	t.mu.Unlock()
	slog.Info("WiFi connected", "to", t.addr)
	go t.receive(conn, done)
	return nil
}

func (t *WiFi) receive(conn net.Conn, done chan struct{}) {
	defer close(done)
	err := t.readFrames(conn)

	t.mu.Lock()
	lost := t.conn == conn && !t.closing
	if lost {
		t.conn = nil
	}
	h, ctx := t.h, t.ctx
	t.mu.Unlock()
	if !lost {
		return // closed on purpose
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		slog.Warn("Device closed the connection", "from", t.addr)
	} else {
		slog.Error("Receive failed", "from", t.addr, "err", err)
	}
	if h.Disconnect != nil {
		h.Disconnect()
	}
	go reconnect(ctx, t.addr, t.retry, t.dial, h)
}

func (t *WiFi) readFrames(conn net.Conn) error {
	header := make([]byte, 2)
	for {
		if _, err := io.ReadFull(conn, header); err != nil {
			return err
		}
		n := int(binary.LittleEndian.Uint16(header))
		if n == 0 {
			continue
		}
		frame := make([]byte, n)
		if _, err := io.ReadFull(conn, frame); err != nil {
			return err
		}
		t.dispatch(frame[0], frame[1:])
	}
}

func (t *WiFi) dispatch(kind byte, payload []byte) {
	switch e := Event(kind); {
	case kind == micAudio:
		if t.h.Audio != nil {
			t.h.Audio(payload)
		}
	case e.valid():
		if t.h.Event != nil {
			t.h.Event(e, payload)
		}
	case kind != speakerAudio:
		slog.Debug("Unknown message type", "type", fmt.Sprintf("0x%02X", kind))
	}
}
