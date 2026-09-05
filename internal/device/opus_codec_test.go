package device

import "testing"

func newOpus(t *testing.T) *opusCodec {
	t.Helper()
	c, err := newOpusCodec(16000, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestOpusProperties(t *testing.T) {
	c := newOpus(t)

	if c.ID() != 2 || c.Type() != Opus || c.SampleRate() != 16000 || c.FrameDurationMs() != 20 {
		t.Fatal("unexpected Opus properties")
	}
	if c.EncodedFrameBytes() != 80 || c.PCMFrameBytes() != 320*2 {
		t.Fatalf("frame bytes = %d/%d", c.EncodedFrameBytes(), c.PCMFrameBytes())
	}
}

func TestOpusEncodeProducesBytes(t *testing.T) {
	c := newOpus(t)
	pcm := make([]byte, c.PCMFrameBytes())

	enc, err := c.Encode(pcm)

	if err != nil || len(enc) == 0 {
		t.Fatalf("Encode = %d bytes, %v", len(enc), err)
	}
}

func TestOpusMusicModeSetsBitrate(t *testing.T) {
	c, err := newOpusCodec(48000, 2, LC3MusicNByte)

	if err != nil || c.Bitrate() != 128000 || c.EncodedFrameBytes() != 160 {
		t.Fatalf("err=%v bitrate=%d frameBytes=%d", err, c.Bitrate(), c.EncodedFrameBytes())
	}
	enc, _ := c.Encode(make([]byte, c.PCMFrameBytes()))
	dec, err := c.Decode(enc)
	if err != nil || len(dec) != c.PCMFrameBytes() {
		t.Fatalf("round trip: %d bytes, %v", len(dec), err)
	}
}

func TestOpusRoundTripPreservesLength(t *testing.T) {
	c := newOpus(t)
	pcm := make([]byte, c.PCMFrameBytes())

	enc, _ := c.Encode(pcm)
	dec, err := c.Decode(enc)

	if err != nil || len(dec) != c.PCMFrameBytes() {
		t.Fatalf("Decode = %d bytes, %v", len(dec), err)
	}
}
