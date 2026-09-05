package device

import (
	"bytes"
	"testing"
)

func TestEventTypeValues(t *testing.T) {
	if EventWakeWord != 0x01 || EventSyncPlay != 0x0B || EventAudioConfig != 0x08 {
		t.Fatal("event values changed")
	}
}

func TestEventTypeString(t *testing.T) {
	if EventTTSStart.String() != "TTS_START" || EventType(0x7F).String() != "EVENT_0x7F" {
		t.Fatal("unexpected String()")
	}
}

func TestEventTypeValid(t *testing.T) {
	if !EventMicConfig.Valid() || EventType(0x20).Valid() {
		t.Fatal("Valid() wrong")
	}
}

func TestAudioConfigMarshal(t *testing.T) {
	cfg := AudioConfig{SampleRate: 16000, EncodedFrameBytes: 40, CodecType: 1, Channels: 2}

	got := cfg.Marshal()

	want := []byte{0x80, 0x3E, 0, 0, 40, 0, 1, 2}
	if !bytes.Equal(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestAudioConfigRoundTrip(t *testing.T) {
	cfg := AudioConfig{SampleRate: 48000, EncodedFrameBytes: 80, CodecType: 2, Channels: 1}

	back, err := UnmarshalAudioConfig(cfg.Marshal())

	if err != nil || back != cfg {
		t.Fatalf("got %+v, %v", back, err)
	}
}

func TestUnmarshalAudioConfigShort(t *testing.T) {
	_, err := UnmarshalAudioConfig([]byte{1, 2})

	if err == nil {
		t.Fatal("expected error")
	}
}
