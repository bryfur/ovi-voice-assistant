package device

import (
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
)

func TestEventNames(t *testing.T) {
	if EventWakeWord != 0x01 || EventSyncPlay != 0x0B || EventTTSStart.String() != "TTS_START" {
		t.Fatal("event values or names changed")
	}
	if Event(0x7F).String() != "EVENT_0x7F" || Event(0x20).valid() || !EventMicConfig.valid() {
		t.Fatal("validity wrong")
	}
}

func TestAudioConfigRoundTrip(t *testing.T) {
	f := codec.Format{Type: codec.LC3, Rate: 48000, Channels: 2, FrameBytes: 80}

	got, err := ParseFormat(AudioConfig(f))

	if err != nil || got != f {
		t.Fatalf("got %+v, %v", got, err)
	}
	if b := AudioConfig(f); b[6] != 1 || b[7] != 2 || len(MicConfig(f)) != 7 {
		t.Fatalf("payload = %v", b)
	}
}

func TestParseFormatMicPayloadAndShort(t *testing.T) {
	f, err := ParseFormat(MicConfig(codec.Format{Type: codec.Opus, Rate: 16000, FrameBytes: 80}))

	if err != nil || f.Type != codec.Opus || f.Rate != 16000 || f.Channels != 1 {
		t.Fatalf("got %+v, %v", f, err)
	}
	if _, err := ParseFormat([]byte{1, 2}); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseWake(t *testing.T) {
	word, score := ParseWake([]byte{0xE8, 0x03, 0x64, 0x00, 'h', 'e', 'y'}) // peak 1000, ambient 100
	legacy, none := ParseWake([]byte("ok"))
	_, zero := ParseWake([]byte{1, 0, 0, 0})

	if word != "hey" || score != 10000 || legacy != "ok" || none != 0 || zero != 1000 {
		t.Fatalf("got %q %d %q %d %d", word, score, legacy, none, zero)
	}
}
