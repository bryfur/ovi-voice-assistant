package music

import (
	"context"
	"encoding/binary"
	"sync"
	"testing"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

type fakeTransport struct {
	mu     sync.Mutex
	events []transport.EventType
	sync   []int64
}

func (f *fakeTransport) Connect() error                                     { return nil }
func (f *fakeTransport) Disconnect() error                                  { return nil }
func (f *fakeTransport) SendAudio([]byte) error                             { return nil }
func (f *fakeTransport) SetEventCallback(transport.EventCallback)           {}
func (f *fakeTransport) SetAudioCallback(transport.AudioCallback)           {}
func (f *fakeTransport) SetDisconnectCallback(transport.DisconnectCallback) {}
func (f *fakeTransport) SetConnectCallback(transport.ConnectCallback)       {}
func (f *fakeTransport) IsConnected() bool                                  { return true }
func (f *fakeTransport) String() string                                     { return "fake" }
func (f *fakeTransport) SendEvent(e transport.EventType, p []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, e)
	if e == transport.EventSyncPlay {
		f.sync = append(f.sync, int64(binary.LittleEndian.Uint64(p)))
	}
	return nil
}

func TestGroupPlayQueuesWithoutStreaming(t *testing.T) {
	g := NewMusicGroup(48000, 2, nil)

	g.Play(context.Background(), []MusicTrack{{Title: "A"}})

	if !g.IsActive() || g.Player.QueueLen() != 1 {
		t.Fatal("Play should queue and activate")
	}
}

func TestGroupResumeStreamsToAllDevicesWithSyncPlay(t *testing.T) {
	g := NewMusicGroup(48000, 2, nil)
	g.Player.ExtractURL = func(context.Context, string) (string, error) { return "", context.Canceled }
	out1, out2 := &captureOutput{}, &captureOutput{}
	tr1, tr2 := &fakeTransport{}, &fakeTransport{}
	g.AddDevice(out1, tr1)
	g.AddDevice(out2, tr2)
	g.Play(context.Background(), []MusicTrack{{Title: "A", VideoID: "x"}})
	before := time.Now().UnixMilli()

	g.Resume(context.Background())
	g.cancelTask() // wait for the stream task to finish

	for _, tr := range []*fakeTransport{tr1, tr2} {
		if len(tr.sync) != 1 || tr.sync[0] < before+SyncBufferMs-50 {
			t.Fatalf("SYNC_PLAY missing or too early: %v", tr.sync)
		}
	}
	for _, out := range []*captureOutput{out1, out2} {
		// Play() first stops (TTS_END), then streaming sends TTS_START … TTS_END.
		start := -1
		for i, e := range out.events {
			if e == transport.EventTTSStart {
				start = i
			}
		}
		if start < 0 || out.events[len(out.events)-1] != transport.EventTTSEnd || len(out.events) < start+2 {
			t.Fatalf("events = %v", out.events)
		}
	}
	if g.DeviceCount() != 2 {
		t.Fatal("device count wrong")
	}
}

func TestGroupPauseAndStopEndPlaybackOnDevices(t *testing.T) {
	g := NewMusicGroup(48000, 2, nil)
	out := &captureOutput{}
	g.AddDevice(out, &fakeTransport{})
	g.Play(context.Background(), []MusicTrack{{Title: "A"}})

	g.Pause(context.Background())
	paused := g.IsActive()
	g.Stop(context.Background())

	if paused || g.Player.QueueLen() != 0 {
		t.Fatalf("paused=%v queue=%d", paused, g.Player.QueueLen())
	}
	if len(out.events) < 2 || out.events[0] != transport.EventTTSEnd {
		t.Fatalf("events = %v", out.events)
	}
}

func TestGroupSkipWithoutRunningTask(t *testing.T) {
	g := NewMusicGroup(48000, 2, nil)
	g.Play(context.Background(), []MusicTrack{{Title: "A"}, {Title: "B"}})

	next := g.Skip()

	if next == nil || next.Title != "B" {
		t.Fatalf("got %+v", next)
	}
}

func TestFanoutSendsToAll(t *testing.T) {
	out1, out2 := &captureOutput{}, &captureOutput{}
	f := &fanoutOutput{outputs: []audioDeviceOutput{out1, out2}}

	f.SendAudio(context.Background(), []byte{1})
	f.SendEvent(context.Background(), transport.EventTTSEnd, nil)

	if out1.totalBytes() != 1 || out2.totalBytes() != 1 || len(out1.events) != 1 || len(out2.events) != 1 {
		t.Fatal("fanout incomplete")
	}
}
