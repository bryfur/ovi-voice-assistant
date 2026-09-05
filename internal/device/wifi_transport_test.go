package device

import (
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

func TestWiFiTransportDefaults(t *testing.T) {
	tr := NewWiFiTransport("192.168.1.42", 0, "")

	if tr.Port() != 6055 || tr.IsConnected() || tr.String() != "192.168.1.42:6055" {
		t.Fatalf("unexpected defaults: %+v", tr)
	}
}

func TestWiFiTransportCustomPort(t *testing.T) {
	tr := NewWiFiTransport("192.168.1.42", 9999, "")

	if tr.Port() != 9999 {
		t.Fatal("custom port not applied")
	}
}

func TestWiFiSendEventNotConnectedIsNoop(t *testing.T) {
	tr := NewWiFiTransport("host", 6055, "")

	err := tr.SendEvent(EventTTSStart, nil)

	if err != nil {
		t.Fatal(err)
	}
}

// fakeDevice is an in-process TCP peer speaking the Ovi framing protocol.
type fakeDevice struct {
	ln   net.Listener
	mu   sync.Mutex
	conn net.Conn
}

func newFakeDevice(t *testing.T) *fakeDevice {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	d := &fakeDevice{ln: ln}
	t.Cleanup(func() { ln.Close() })
	return d
}

func (d *fakeDevice) accept(t *testing.T) net.Conn {
	t.Helper()
	conn, err := d.ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	d.mu.Lock()
	d.conn = conn
	d.mu.Unlock()
	return conn
}

func writeFrame(conn net.Conn, payload []byte) {
	buf := make([]byte, 2+len(payload))
	binary.LittleEndian.PutUint16(buf, uint16(len(payload)))
	copy(buf[2:], payload)
	conn.Write(buf)
}

func readFrame(t *testing.T, conn net.Conn) []byte {
	t.Helper()
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	hdr := make([]byte, 2)
	if _, err := io.ReadFull(conn, hdr); err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, binary.LittleEndian.Uint16(hdr))
	if _, err := io.ReadFull(conn, payload); err != nil {
		t.Fatal(err)
	}
	return payload
}

func TestWiFiSendsLengthPrefixedFrames(t *testing.T) {
	dev := newFakeDevice(t)
	tr := NewWiFiTransport("127.0.0.1", dev.ln.Addr().(*net.TCPAddr).Port, "")
	if err := tr.Connect(); err != nil {
		t.Fatal(err)
	}
	defer tr.Disconnect()
	conn := dev.accept(t)

	tr.SendEvent(EventTTSStart, []byte{9})
	tr.SendAudio([]byte{1, 2, 3})

	if got := readFrame(t, conn); got[0] != byte(EventTTSStart) || got[1] != 9 {
		t.Fatalf("event frame = %v", got)
	}
	if got := readFrame(t, conn); got[0] != speakerAudioType || len(got) != 4 {
		t.Fatalf("audio frame = %v", got)
	}
}

func TestWiFiDispatchesMicAudioAndEvents(t *testing.T) {
	dev := newFakeDevice(t)
	tr := NewWiFiTransport("127.0.0.1", dev.ln.Addr().(*net.TCPAddr).Port, "")
	audio := make(chan []byte, 1)
	events := make(chan EventType, 1)
	tr.SetAudioCallback(func(b []byte) { audio <- b })
	tr.SetEventCallback(func(e EventType, p []byte) { events <- e })
	if err := tr.Connect(); err != nil {
		t.Fatal(err)
	}
	defer tr.Disconnect()
	conn := dev.accept(t)

	writeFrame(conn, []byte{micAudioType, 7, 8})
	writeFrame(conn, []byte{byte(EventWakeWord)})
	writeFrame(conn, []byte{0x7F}) // unknown → ignored

	select {
	case got := <-audio:
		if len(got) != 2 || got[0] != 7 {
			t.Fatalf("audio = %v", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no audio callback")
	}
	select {
	case e := <-events:
		if e != EventWakeWord {
			t.Fatalf("event = %v", e)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no event callback")
	}
}

func TestWiFiDisconnectCallbackAndReconnect(t *testing.T) {
	dev := newFakeDevice(t)
	tr := NewWiFiTransport("127.0.0.1", dev.ln.Addr().(*net.TCPAddr).Port, "")
	tr.reconnectDelay = 50 * time.Millisecond
	disconnected := make(chan struct{}, 1)
	reconnected := make(chan struct{}, 1)
	tr.SetDisconnectCallback(func() { disconnected <- struct{}{} })
	tr.SetConnectCallback(func() { reconnected <- struct{}{} })
	if err := tr.Connect(); err != nil {
		t.Fatal(err)
	}
	defer tr.Disconnect()
	conn := dev.accept(t)

	conn.Close() // device drops the connection

	select {
	case <-disconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("no disconnect callback")
	}
	dev.accept(t)
	select {
	case <-reconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("no reconnect callback")
	}
	if !tr.IsConnected() {
		t.Fatal("should be connected after reconnect")
	}
}
