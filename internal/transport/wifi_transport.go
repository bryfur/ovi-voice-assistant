package transport

import (
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

// TCP framing constants.
const (
	MicAudioType     = 0x20
	SpeakerAudioType = 0x21
	// ReconnectDelay is the pause between reconnect attempts.
	ReconnectDelay = 3 * time.Second
)

// WiFiTransport is a transport over WiFi using a direct TCP connection.
//
// The Ovi server is the TCP client. The ESPHome device is the TCP server
// listening on port 6055. Communication uses a simple length-prefix
// binary protocol.
//
// TCP frame format:
//
//	[2 bytes LE: payload length][payload bytes]
//
// Payload types:
//
//	Control event:  [1 byte event_type][event payload]
//	Mic audio:      [0x20][codec frame bytes]   (device -> server)
//	Speaker audio:  [0x21][codec frame bytes]   (server -> device)
type WiFiTransport struct {
	host          string
	port          int
	encryptionKey string // reserved for future Noise handshake

	mu        sync.Mutex
	conn      net.Conn
	connected bool
	stopping  bool
	recvDone  chan struct{}
	reconnect chan struct{}

	eventCB      EventCallback
	audioCB      AudioCallback
	disconnectCB DisconnectCallback
	connectCB    ConnectCallback

	// Dial allows tests to inject a dialer. Defaults to net.Dialer with keepalive.
	Dial func(addr string) (net.Conn, error)

	reconnectDelay time.Duration
}

// NewWiFiTransport creates a transport for host:port.
func NewWiFiTransport(host string, port int, encryptionKey string) *WiFiTransport {
	if port == 0 {
		port = 6055
	}
	t := &WiFiTransport{
		host:           host,
		port:           port,
		encryptionKey:  encryptionKey,
		reconnectDelay: ReconnectDelay,
	}
	t.Dial = t.defaultDial
	return t
}

// Host returns the configured host.
func (t *WiFiTransport) Host() string { return t.host }

// Port returns the configured port.
func (t *WiFiTransport) Port() int { return t.port }

// String implements DeviceTransport.
func (t *WiFiTransport) String() string {
	return net.JoinHostPort(t.host, strconv.Itoa(t.port))
}

// IsConnected implements DeviceTransport.
func (t *WiFiTransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected
}

func (t *WiFiTransport) defaultDial(addr string) (net.Conn, error) {
	// Enable TCP keepalive so we detect a dead peer (e.g. device power
	// loss) within ~25s instead of waiting for the OS TCP timeout.
	d := net.Dialer{
		Timeout: 10 * time.Second,
		KeepAliveConfig: net.KeepAliveConfig{
			Enable:   true,
			Idle:     10 * time.Second,
			Interval: 5 * time.Second,
			Count:    3,
		},
	}
	return d.Dial("tcp", addr)
}

// SetEventCallback implements DeviceTransport.
func (t *WiFiTransport) SetEventCallback(cb EventCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.eventCB = cb
}

// SetAudioCallback implements DeviceTransport.
func (t *WiFiTransport) SetAudioCallback(cb AudioCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.audioCB = cb
}

// SetDisconnectCallback implements DeviceTransport.
func (t *WiFiTransport) SetDisconnectCallback(cb DisconnectCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.disconnectCB = cb
}

// SetConnectCallback implements DeviceTransport.
func (t *WiFiTransport) SetConnectCallback(cb ConnectCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connectCB = cb
}

// Connect implements DeviceTransport.
func (t *WiFiTransport) Connect() error {
	t.mu.Lock()
	t.stopping = false
	t.mu.Unlock()
	return t.establish()
}

// Disconnect implements DeviceTransport.
func (t *WiFiTransport) Disconnect() error {
	t.mu.Lock()
	t.stopping = true
	t.connected = false
	if t.reconnect != nil {
		close(t.reconnect)
		t.reconnect = nil
	}
	t.mu.Unlock()
	t.closeConn()
	return nil
}

// SendEvent implements DeviceTransport.
func (t *WiFiTransport) SendEvent(event EventType, payload []byte) error {
	if !t.IsConnected() {
		slog.Warn("Cannot send event -- not connected", "event", event.String())
		return nil
	}
	frame := make([]byte, 0, 1+len(payload))
	frame = append(frame, byte(event))
	frame = append(frame, payload...)
	return t.sendFrame(frame)
}

// SendAudio implements DeviceTransport.
func (t *WiFiTransport) SendAudio(data []byte) error {
	if !t.IsConnected() {
		slog.Warn("Cannot send audio -- not connected")
		return nil
	}
	frame := make([]byte, 0, 1+len(data))
	frame = append(frame, SpeakerAudioType)
	frame = append(frame, data...)
	return t.sendFrame(frame)
}

// -- Internal: connection management --

func (t *WiFiTransport) establish() error {
	addr := t.String()
	slog.Info("WiFi connecting", "addr", addr)
	conn, err := t.Dial(addr)
	if err != nil {
		return err
	}
	t.mu.Lock()
	t.conn = conn
	t.connected = true
	t.recvDone = make(chan struct{})
	done := t.recvDone
	t.mu.Unlock()
	slog.Info("WiFi connected", "addr", addr)

	// TODO: Noise handshake when encryptionKey is set

	go t.recvLoop(conn, done)
	return nil
}

func (t *WiFiTransport) closeConn() {
	t.mu.Lock()
	conn := t.conn
	done := t.recvDone
	t.conn = nil
	t.recvDone = nil
	t.mu.Unlock()
	if conn != nil {
		_ = conn.Close()
	}
	if done != nil {
		<-done
	}
}

// sendFrame writes a length-prefixed frame over TCP.
func (t *WiFiTransport) sendFrame(data []byte) error {
	if len(data) > 0xFFFF {
		return fmt.Errorf("frame too large: %d bytes", len(data))
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn == nil {
		return nil
	}
	buf := make([]byte, 2+len(data))
	binary.LittleEndian.PutUint16(buf, uint16(len(data)))
	copy(buf[2:], data)
	if _, err := t.conn.Write(buf); err != nil {
		slog.Error("Failed to send frame", "err", err)
		return err
	}
	return nil
}

// -- Internal: receive loop --

func (t *WiFiTransport) recvLoop(conn net.Conn, done chan struct{}) {
	defer close(done)
	header := make([]byte, 2)
	var readErr error
	for {
		if _, err := io.ReadFull(conn, header); err != nil {
			readErr = err
			break
		}
		length := int(binary.LittleEndian.Uint16(header))
		if length == 0 {
			continue
		}
		payload := make([]byte, length)
		if _, err := io.ReadFull(conn, payload); err != nil {
			readErr = err
			break
		}
		t.dispatch(payload)
	}

	t.mu.Lock()
	wasConnected := t.connected && t.conn == conn
	t.connected = false
	stopping := t.stopping
	disconnectCB := t.disconnectCB
	t.mu.Unlock()

	if !wasConnected {
		return // closed deliberately (Disconnect/reconnect)
	}
	if errors.Is(readErr, io.EOF) || errors.Is(readErr, io.ErrUnexpectedEOF) {
		slog.Warn("WiFi connection closed by device", "host", t.host)
	} else if readErr != nil && !stopping {
		slog.Error("WiFi receive loop error", "host", t.host, "err", readErr)
	}

	if disconnectCB != nil {
		disconnectCB()
	}
	if !stopping {
		t.mu.Lock()
		stop := make(chan struct{})
		t.reconnect = stop
		t.mu.Unlock()
		go t.reconnectLoop(stop)
	}
}

func (t *WiFiTransport) dispatch(payload []byte) {
	msgType := payload[0]
	msgData := payload[1:]
	t.mu.Lock()
	audioCB := t.audioCB
	eventCB := t.eventCB
	t.mu.Unlock()

	switch msgType {
	case MicAudioType:
		if audioCB != nil {
			audioCB(msgData)
		}
	case SpeakerAudioType:
		// We don't receive speaker audio from the device.
	default:
		event := EventType(msgType)
		if !event.Valid() {
			slog.Debug("Unknown event type", "type", fmt.Sprintf("0x%02X", msgType))
			return
		}
		if eventCB != nil {
			eventCB(event, msgData)
		}
	}
}

func (t *WiFiTransport) reconnectLoop(stop chan struct{}) {
	for {
		select {
		case <-stop:
			return
		case <-time.After(t.reconnectDelay):
		}
		t.mu.Lock()
		stopping := t.stopping
		t.mu.Unlock()
		if stopping {
			return
		}
		slog.Info("WiFi reconnecting", "addr", t.String())
		t.closeConn()
		if err := t.establish(); err != nil {
			slog.Warn("WiFi reconnect failed, retrying", "host", t.host, "delay", t.reconnectDelay, "err", err)
			continue
		}
		slog.Info("WiFi reconnected", "addr", t.String())
		t.mu.Lock()
		if t.reconnect == stop {
			t.reconnect = nil
		}
		cb := t.connectCB
		t.mu.Unlock()
		if cb != nil {
			cb()
		}
		return
	}
}
