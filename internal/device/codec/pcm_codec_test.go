package codec

import (
	"bytes"
	"testing"
)

func TestPCMCodecProperties(t *testing.T) {
	c := NewPCMCodec(16000, 1)

	if c.ID() != 0 || c.Type() != PCM || c.SampleRate() != 16000 || c.FrameDurationMs() != 20 {
		t.Fatal("unexpected PCM codec properties")
	}
	if c.PCMFrameBytes() != 640 || c.EncodedFrameBytes() != 640 {
		t.Fatalf("frame bytes = %d/%d", c.PCMFrameBytes(), c.EncodedFrameBytes())
	}
}

func TestPCMCodecStereo48k(t *testing.T) {
	c := NewPCMCodec(48000, 2)

	got := c.PCMFrameBytes()

	if got != 3840 {
		t.Fatalf("PCMFrameBytes = %d, want 3840", got)
	}
}

func TestPCMCodecIdentity(t *testing.T) {
	c := NewPCMCodec(16000, 1)
	data := []byte{1, 2, 3, 4}

	enc, _ := c.Encode(data)
	dec, _ := c.Decode(data)

	if !bytes.Equal(enc, data) || !bytes.Equal(dec, data) {
		t.Fatal("PCM codec must be identity")
	}
}
