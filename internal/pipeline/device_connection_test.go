package pipeline

import (
	"context"
	"encoding/binary"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// fakeTransport records outbound traffic and lets tests inject inbound events.
type fakeTransport struct {
	mu         sync.Mutex
	events     []device.EventType
	payloads   [][]byte
	audio      [][]byte
	eventCB    device.EventCallback
	audioCB    device.AudioCallback
	connectCB  device.ConnectCallback
	disconnect device.DisconnectCallback
	connected  bool
}

func (f *fakeTransport) Connect() error    { f.connected = true; return nil }
func (f *fakeTransport) Disconnect() error { f.connected = false; return nil }
func (f *fakeTransport) SendEvent(e device.EventType, p []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, e)
	f.payloads = append(f.payloads, append([]byte(nil), p...))
	return nil
}
func (f *fakeTransport) SendAudio(d []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.audio = append(f.audio, d)
	return nil
}
func (f *fakeTransport) SetEventCallback(cb device.EventCallback)           { f.eventCB = cb }
func (f *fakeTransport) SetAudioCallback(cb device.AudioCallback)           { f.audioCB = cb }
func (f *fakeTransport) SetDisconnectCallback(cb device.DisconnectCallback) { f.disconnect = cb }
func (f *fakeTransport) SetConnectCallback(cb device.ConnectCallback)       { f.connectCB = cb }
func (f *fakeTransport) IsConnected() bool                                  { return f.connected }
func (f *fakeTransport) String() string                                     { return "fake" }
func (f *fakeTransport) eventList() []device.EventType {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]device.EventType(nil), f.events...)
}
func (f *fakeTransport) hasEvent(e device.EventType) bool {
	for _, got := range f.eventList() {
		if got == e {
			return true
		}
	}
	return false
}

// fakePipeline records runs; Run drains the mic until ctx ends or a nil sentinel.
type fakePipeline struct {
	mu        sync.Mutex
	runs      int
	announces []string
	resets    int
	mic       [][]byte
	followUp  bool
	started   chan struct{}
	block     bool
}

func newFakePipeline() *fakePipeline {
	return &fakePipeline{started: make(chan struct{}, 16)}
}

func (p *fakePipeline) Run(ctx context.Context, _ device.Output, mic <-chan []byte, _ *agent.Context) bool {
	p.mu.Lock()
	p.runs++
	p.mu.Unlock()
	p.started <- struct{}{}
	if !p.block {
		// Grab whatever audio is immediately available.
		for {
			select {
			case chunk := <-mic:
				p.mu.Lock()
				p.mic = append(p.mic, chunk)
				p.mu.Unlock()
				continue
			default:
			}
			break
		}
		return p.followUp
	}
	for {
		select {
		case <-ctx.Done():
			return false
		case chunk := <-mic:
			p.mu.Lock()
			p.mic = append(p.mic, chunk)
			p.mu.Unlock()
		}
	}
}

func (p *fakePipeline) Announce(_ context.Context, _ device.Output, text string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.announces = append(p.announces, text)
}

func (p *fakePipeline) ResetHistory() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.resets++
}

func (p *fakePipeline) runCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.runs
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("condition not met in time")
}

func newConn(t *testing.T, opts DeviceOptions) (*DeviceConnection, *fakeTransport, *fakePipeline) {
	t.Helper()
	tr := &fakeTransport{}
	pl := newFakePipeline()
	s := config.Default()
	s.Transport.Codec = "pcm"
	c := NewDeviceConnection(tr, codec.NewPCMCodec(16000, 1), pl, s, opts)
	c.SetupDelay = 0
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Stop() })
	return c, tr, pl
}

func TestStartSendsAudioConfigAndBuildsContext(t *testing.T) {
	c, tr, _ := newConn(t, DeviceOptions{Name: "dev1"})

	events := tr.eventList()

	if len(events) != 1 || events[0] != device.EventAudioConfig || !tr.connected {
		t.Fatalf("events = %v", events)
	}
	cfg, _ := device.UnmarshalAudioConfig(tr.payloads[0])
	if cfg.SampleRate != 16000 || cfg.CodecType != 0 {
		t.Fatalf("config = %+v", cfg)
	}
	if c.Context() == nil || c.Context().MusicPlayer == nil || c.Name != "dev1" {
		t.Fatal("context not built")
	}
}

func TestSetScheduler(t *testing.T) {
	c, _, _ := newConn(t, DeviceOptions{})

	c.SetScheduler(nil)

	if c.Context().Scheduler != nil {
		t.Fatal("nil setter should clear")
	}
}

func TestWakeWordStartsPipelineWithoutArbitration(t *testing.T) {
	c, tr, pl := newConn(t, DeviceOptions{})
	pl.block = true
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint16(payload[0:], 2000)
	binary.LittleEndian.PutUint16(payload[2:], 100)
	payload = append(payload, []byte("okay nabu")...)

	tr.eventCB(device.EventWakeWord, payload)
	<-pl.started
	tr.audioCB([]byte{1, 2, 3, 4})
	waitFor(t, func() bool {
		pl.mu.Lock()
		defer pl.mu.Unlock()
		return len(pl.mic) == 1
	})

	if pl.resets != 1 || !c.TaskRunning() {
		t.Fatalf("resets=%d running=%v", pl.resets, c.TaskRunning())
	}
}

func TestWakeWordUsesArbitrationCallback(t *testing.T) {
	var gotScore int
	var gotWord string
	var gotConn *DeviceConnection
	c, tr, pl := newConn(t, DeviceOptions{OnWake: func(conn *DeviceConnection, score int, word string) {
		gotConn, gotScore, gotWord = conn, score, word
	}})
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint16(payload[0:], 3000)
	binary.LittleEndian.PutUint16(payload[2:], 1500)
	payload = append(payload, []byte("hey jarvis")...)

	tr.eventCB(device.EventWakeWord, payload)

	if gotConn != c || gotScore != 2000 || gotWord != "hey jarvis" || pl.runCount() != 0 {
		t.Fatalf("conn=%v score=%d word=%q runs=%d", gotConn == c, gotScore, gotWord, pl.runCount())
	}
}

func TestWakeWordLegacyPayload(t *testing.T) {
	var gotWord string
	var gotScore int
	_, tr, _ := newConn(t, DeviceOptions{OnWake: func(_ *DeviceConnection, score int, word string) { gotScore, gotWord = score, word }})

	tr.eventCB(device.EventWakeWord, []byte("ok"))

	if gotWord != "ok" || gotScore != 0 {
		t.Fatalf("word=%q score=%d", gotWord, gotScore)
	}
}

func TestStartPipelineAndAbortWake(t *testing.T) {
	c, tr, pl := newConn(t, DeviceOptions{OnWake: func(*DeviceConnection, int, string) {}})
	pl.block = true
	tr.eventCB(device.EventWakeWord, []byte("w"))
	tr.audioCB([]byte{9}) // buffered during the arbitration window

	c.StartPipeline("w")
	<-pl.started
	waitFor(t, func() bool {
		pl.mu.Lock()
		defer pl.mu.Unlock()
		return len(pl.mic) == 1
	})
	c.AbortWake()

	if !tr.hasEvent(device.EventWakeAbort) || c.TaskRunning() || pl.resets != 1 {
		t.Fatalf("abort=%v running=%v resets=%d", tr.hasEvent(device.EventWakeAbort), c.TaskRunning(), pl.resets)
	}
}

func TestFollowUpSkipsArbitration(t *testing.T) {
	c, tr, pl := newConn(t, DeviceOptions{OnWake: func(*DeviceConnection, int, string) { t.Fatal("arbitration must be skipped") }})
	pl.followUp = true
	c.StartPipeline("w") // first session reports follow-up
	<-pl.started
	waitFor(t, func() bool { return !c.TaskRunning() })

	tr.eventCB(device.EventWakeWord, []byte("w")) // follow-up wake
	<-pl.started
	waitFor(t, func() bool { return !c.TaskRunning() })

	if pl.runCount() != 2 {
		t.Fatalf("runs = %d", pl.runCount())
	}
}

func TestMicConfigRequestsPreferredCodec(t *testing.T) {
	c, tr, _ := newConn(t, DeviceOptions{})
	payload := make([]byte, 7)
	binary.LittleEndian.PutUint32(payload[0:], 16000)
	binary.LittleEndian.PutUint16(payload[4:], 40)
	payload[6] = 1 // device reports LC3, server prefers PCM

	tr.eventCB(device.EventMicConfig, payload)

	events := tr.eventList()
	if events[len(events)-1] != device.EventMicConfig {
		t.Fatalf("expected MIC_CONFIG request, events = %v", events)
	}
	req := tr.payloads[len(tr.payloads)-1]
	if binary.LittleEndian.Uint32(req[0:]) != 16000 || req[6] != 0 {
		t.Fatalf("request = %v", req)
	}
	c.mu.Lock()
	mic := c.micCodec
	c.mu.Unlock()
	if mic == nil || mic.Type() != codec.PCM {
		t.Fatalf("mic codec = %v", mic)
	}
}

func TestMicConfigMatchingCodecNoRequest(t *testing.T) {
	_, tr, _ := newConn(t, DeviceOptions{})
	payload := make([]byte, 7)
	binary.LittleEndian.PutUint32(payload[0:], 16000)
	payload[6] = 0 // PCM, same as server preference

	tr.eventCB(device.EventMicConfig, payload)

	if tr.hasEvent(device.EventMicConfig) {
		t.Fatal("no MIC_CONFIG request expected when codecs match")
	}
}

func TestAnnounceRunsPipelineAnnounce(t *testing.T) {
	c, _, pl := newConn(t, DeviceOptions{})

	err := c.AnnounceAndWait(context.Background(), "Timer done.")

	if err != nil || len(pl.announces) != 1 || pl.announces[0] != "Timer done." {
		t.Fatalf("err=%v announces=%v", err, pl.announces)
	}
}

func TestAnnounceInterruptsRunningSession(t *testing.T) {
	c, _, pl := newConn(t, DeviceOptions{})
	pl.block = true
	c.StartPipeline("w")
	<-pl.started

	c.AnnounceAndWait(context.Background(), "Interrupt.")

	if len(pl.announces) != 1 {
		t.Fatal("announce not run")
	}
}

func TestDisconnectCancelsTask(t *testing.T) {
	c, tr, pl := newConn(t, DeviceOptions{})
	pl.block = true
	c.StartPipeline("w")
	<-pl.started

	tr.disconnect()
	waitFor(t, func() bool { return !c.TaskRunning() })
}

func TestReconnectResendsAudioConfig(t *testing.T) {
	_, tr, _ := newConn(t, DeviceOptions{})

	tr.connectCB()

	events := tr.eventList()
	if len(events) != 2 || events[1] != device.EventAudioConfig {
		t.Fatalf("events = %v", events)
	}
}
