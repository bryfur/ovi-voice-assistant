package device

import (
	"math"
	"testing"
)

func newLC3(t *testing.T) *lc3Codec {
	t.Helper()
	c, err := newLC3Codec(16000, 1, lc3DefaultNByte)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

func TestLC3Properties(t *testing.T) {
	c := newLC3(t)

	if c.ID() != 1 || c.Type() != LC3 || c.SampleRate() != 16000 || c.FrameDurationMs() != 10 {
		t.Fatal("unexpected LC3 properties")
	}
	if c.EncodedFrameBytes() != 40 || c.PCMFrameBytes() != 160*2 {
		t.Fatalf("frame bytes = %d/%d", c.EncodedFrameBytes(), c.PCMFrameBytes())
	}
}

func TestLC3EncodeProducesBytes(t *testing.T) {
	c := newLC3(t)
	pcm := make([]byte, c.PCMFrameBytes())

	enc, err := c.Encode(pcm)

	if err != nil || len(enc) != c.EncodedFrameBytes() {
		t.Fatalf("Encode = %d bytes, %v", len(enc), err)
	}
}

func TestLC3RoundTripPreservesLength(t *testing.T) {
	c := newLC3(t)
	pcm := make([]byte, c.PCMFrameBytes())

	enc, _ := c.Encode(pcm)
	dec, err := c.Decode(enc)

	if err != nil || len(dec) != c.PCMFrameBytes() {
		t.Fatalf("Decode = %d bytes, %v", len(dec), err)
	}
}

func TestLC3StereoWireFrameSize(t *testing.T) {
	c, err := newLC3Codec(48000, 2, LC3MusicNByte)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	pcm := make([]byte, c.PCMFrameBytes())

	enc, err := c.Encode(pcm)

	if err != nil || len(enc) != 160 || c.EncodedFrameBytes() != LC3MusicNByte {
		t.Fatalf("stereo music Encode = %d bytes (per channel %d), %v", len(enc), c.EncodedFrameBytes(), err)
	}
	if dec, err := c.Decode(enc); err != nil || len(dec) != c.PCMFrameBytes() {
		t.Fatalf("stereo music Decode = %d bytes, %v", len(dec), err)
	}
}

func TestLC3RoundTripSineHasEnergy(t *testing.T) {
	c := newLC3(t)
	samples := c.PCMFrameBytes() / 2
	pcm := make([]byte, c.PCMFrameBytes())
	for i := 0; i < samples; i++ {
		v := int16(8000 * math.Sin(2*math.Pi*440*float64(i)/16000))
		pcm[i*2] = byte(v)
		pcm[i*2+1] = byte(v >> 8)
	}

	var dec []byte
	for i := 0; i < 5; i++ { // let the codec warm up past its algorithmic delay
		enc, _ := c.Encode(pcm)
		dec, _ = c.Decode(enc)
	}

	var energy float64
	for i := 0; i < samples; i++ {
		v := float64(int16(uint16(dec[i*2]) | uint16(dec[i*2+1])<<8))
		energy += v * v
	}
	if energy == 0 {
		t.Fatal("decoded audio is silent")
	}
}
