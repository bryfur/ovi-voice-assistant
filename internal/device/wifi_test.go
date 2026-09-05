package device

import (
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"
)

// peer is an in-process ESPHome device speaking the framing protocol.
type peer struct{ ln net.Listener }

func listen(t *testing.T) (*peer, *WiFi) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	tr := NewWiFi("127.0.0.1", ln.Addr().(*net.TCPAddr).Port)
	tr.retry = 50 * time.Millisecond
	t.Cleanup(func() { tr.Disconnect() })
	return &peer{ln}, tr
}

func (p *peer) accept(t *testing.T) net.Conn {
	t.Helper()
	c, err := p.ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func write(c net.Conn, payload []byte) {
	buf := make([]byte, 2+len(payload))
	binary.LittleEndian.PutUint16(buf, uint16(len(payload)))
	copy(buf[2:], payload)
	c.Write(buf)
}

func read(t *testing.T, c net.Conn) []byte {
	t.Helper()
	c.SetReadDeadline(time.Now().Add(2 * time.Second))
	hdr := make([]byte, 2)
	if _, err := io.ReadFull(c, hdr); err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, binary.LittleEndian.Uint16(hdr))
	if _, err := io.ReadFull(c, payload); err != nil {
		t.Fatal(err)
	}
	return payload
}

func TestWiFiDefaultPortAndString(t *testing.T) {
	if NewWiFi("192.168.1.42", 0).String() != "192.168.1.42:6055" || NewWiFi("h", 9999).String() != "h:9999" {
		t.Fatal("address formatting wrong")
	}
}

func TestWiFiSendsFramesAndDropsWhenDisconnected(t *testing.T) {
	p, tr := listen(t)
	if err := tr.SendEvent(EventTTSStart, nil); err != nil {
		t.Fatal("sending while disconnected must be a no-op")
	}
	if err := tr.Connect(Handler{}); err != nil {
		t.Fatal(err)
	}
	c := p.accept(t)

	tr.SendEvent(EventTTSStart, []byte{9})
	tr.SendAudio([]byte{1, 2, 3})

	if got := read(t, c); got[0] != byte(EventTTSStart) || got[1] != 9 {
		t.Fatalf("event frame = %v", got)
	}
	if got := read(t, c); got[0] != speakerAudio || len(got) != 4 {
		t.Fatalf("audio frame = %v", got)
	}
}

func TestWiFiDispatchesMicAudioAndEvents(t *testing.T) {
	p, tr := listen(t)
	audio, events := make(chan []byte, 1), make(chan Event, 1)
	tr.Connect(Handler{Audio: func(b []byte) { audio <- b }, Event: func(e Event, _ []byte) { events <- e }})
	c := p.accept(t)

	write(c, []byte{micAudio, 7, 8})
	write(c, []byte{byte(EventWakeWord)})
	write(c, []byte{0x7F}) // unknown, ignored

	select {
	case got := <-audio:
		if len(got) != 2 || got[0] != 7 {
			t.Fatalf("audio = %v", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no audio")
	}
	select {
	case e := <-events:
		if e != EventWakeWord {
			t.Fatalf("event = %v", e)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no event")
	}
}

func TestWiFiReconnectsAfterDrop(t *testing.T) {
	p, tr := listen(t)
	dropped, back := make(chan struct{}, 1), make(chan struct{}, 1)
	tr.Connect(Handler{Disconnect: func() { dropped <- struct{}{} }, Connect: func() { back <- struct{}{} }})
	c := p.accept(t)

	c.Close() // the device goes away

	select {
	case <-dropped:
	case <-time.After(2 * time.Second):
		t.Fatal("no disconnect callback")
	}
	c2 := p.accept(t)
	select {
	case <-back:
	case <-time.After(2 * time.Second):
		t.Fatal("no reconnect callback")
	}
	tr.SendEvent(EventTTSEnd, nil)
	if got := read(t, c2); got[0] != byte(EventTTSEnd) {
		t.Fatalf("frame after reconnect = %v", got)
	}
}
