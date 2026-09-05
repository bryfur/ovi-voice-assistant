package pipeline

import (
	"encoding/binary"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

func newConn(t *testing.T, player *music.Player) (*Connection, *fakeTransport, *fakeVoice) {
	t.Helper()
	tr, v := &fakeTransport{}, newFakeVoice()
	c, err := NewConnection(tr, "pcm", v, player)
	if err != nil {
		t.Fatal(err)
	}
	c.SetupDelay = 0
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Stop() })
	return c, tr, v
}

func wakePayload(peak, ambient uint16, word string) []byte {
	p := binary.LittleEndian.AppendUint16(nil, peak)
	p = binary.LittleEndian.AppendUint16(p, ambient)
	return append(p, word...)
}

func TestStartConfiguresDeviceAndBuildsEnv(t *testing.T) {
	c, tr, _ := newConn(t, nil)

	events := tr.sent()

	f, _ := device.ParseFormat(tr.lastPayload())
	if len(events) != 1 || events[0] != device.EventAudioConfig || f.Rate != 16000 || f.Type != codec.PCM {
		t.Fatalf("events=%v format=%+v", events, f)
	}
	if c.Env() == nil || c.Env().Announce == nil || c.Name != "fake" {
		t.Fatal("env not built")
	}
	c.SetScheduler(nil)
}

func TestWakeStartsSessionWithoutArbitration(t *testing.T) {
	c, tr, v := newConn(t, nil)
	v.block = true

	tr.h.Event(device.EventWakeWord, wakePayload(2000, 100, "okay nabu"))
	<-v.started
	tr.h.Audio([]byte{1, 2, 3, 4})
	waitFor(t, func() bool { _, _, mic := v.count(); return mic == 1 })

	if _, resets, _ := v.count(); resets != 1 || !c.Busy() {
		t.Fatalf("resets=%d busy=%v", resets, c.Busy())
	}
}

func TestWakeAsksOnWakeToArbitrate(t *testing.T) {
	var got struct {
		conn  *Connection
		score int
		word  string
	}
	c, tr, v := newConn(t, nil)
	c.OnWake = func(conn *Connection, score int, word string) { got.conn, got.score, got.word = conn, score, word }

	tr.h.Event(device.EventWakeWord, wakePayload(3000, 1500, "hey jarvis"))
	tr.h.Event(device.EventWakeWord, []byte("ok")) // older firmware: word only

	if runs, _, _ := v.count(); got.conn != c || got.score != 0 || got.word != "ok" || runs != 0 {
		t.Fatalf("got %+v runs=%d", got, runs)
	}
}

func TestStartSessionAndAbortWake(t *testing.T) {
	c, tr, v := newConn(t, nil)
	c.OnWake = func(*Connection, int, string) {}
	v.block = true
	tr.h.Event(device.EventWakeWord, []byte("w"))
	tr.h.Audio([]byte{9}) // heard during the arbitration window

	c.StartSession("w")
	<-v.started
	waitFor(t, func() bool { _, _, mic := v.count(); return mic == 1 })
	c.AbortWake()

	if _, resets, _ := v.count(); !tr.has(device.EventWakeAbort) || c.Busy() || resets != 1 {
		t.Fatalf("abort=%v busy=%v resets=%d", tr.has(device.EventWakeAbort), c.Busy(), resets)
	}
}

func TestFollowUpSkipsArbitration(t *testing.T) {
	c, tr, v := newConn(t, nil)
	c.OnWake = func(*Connection, int, string) { t.Error("arbitration must be skipped") }
	v.followUp = true
	c.StartSession("w") // the reply asks a question
	<-v.started
	waitFor(t, func() bool { return !c.Busy() })

	tr.h.Event(device.EventWakeWord, []byte("w")) // the answer
	<-v.started
	waitFor(t, func() bool { return !c.Busy() })

	if runs, _, _ := v.count(); runs != 2 {
		t.Fatalf("runs = %d", runs)
	}
}

func TestMicConfigRequestsPreferredCodec(t *testing.T) {
	c, tr, _ := newConn(t, nil)
	lc3 := device.MicConfig(codec.Format{Type: codec.LC3, Rate: 16000, FrameBytes: 40})
	pcm := device.MicConfig(codec.Format{Type: codec.PCM, Rate: 16000})

	tr.h.Event(device.EventMicConfig, lc3) // the server prefers PCM
	requested := tr.has(device.EventMicConfig)
	f, _ := device.ParseFormat(tr.lastPayload())
	c.mu.Lock()
	mic := c.mic
	c.mu.Unlock()

	if !requested || f.Type != codec.PCM || f.Rate != 16000 || mic.Format().Type != codec.PCM {
		t.Fatalf("requested=%v format=%+v mic=%v", requested, f, mic)
	}
	before := len(tr.sent())
	tr.h.Event(device.EventMicConfig, pcm) // already what we want
	if len(tr.sent()) != before {
		t.Fatal("no request expected when the codecs match")
	}
}

func TestAnnounceInterruptsSession(t *testing.T) {
	c, _, v := newConn(t, nil)
	v.block = true
	c.StartSession("w")
	<-v.started

	c.Announce("Timer done.")
	waitFor(t, func() bool { return !c.Busy() })

	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.announces) != 1 || v.announces[0] != "Timer done." {
		t.Fatalf("announces = %v", v.announces)
	}
}

func TestDisconnectCancelsAndReconnectReconfigures(t *testing.T) {
	c, tr, v := newConn(t, nil)
	v.block = true
	c.StartSession("w")
	<-v.started

	tr.h.Disconnect()
	waitFor(t, func() bool { return !c.Busy() })
	tr.h.Connect()

	if events := tr.sent(); len(events) != 2 || events[1] != device.EventAudioConfig {
		t.Fatalf("events = %v", events)
	}
}

func TestSessionPausesAndResumesMusic(t *testing.T) {
	svc := &silentService{started: make(chan struct{}, 8)}
	player := music.NewPlayer(map[string]music.Service{"fake": svc})
	c, tr, v := newConn(t, player)
	player.Play([]music.Track{{Title: "A", Service: "fake"}})
	player.Continue()
	<-svc.started

	tr.h.Event(device.EventWakeWord, []byte("w")) // interrupts music
	<-v.started
	waitFor(t, func() bool { return !c.Busy() })
	<-svc.started // music continued after the session

	if !tr.has(device.EventTTSEnd) || c.Env().Music != player {
		t.Fatalf("events = %v", tr.sent())
	}
	player.Stop()
}
