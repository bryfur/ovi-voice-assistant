package device

// PCMCodec is an identity codec that passes PCM audio through unchanged.
type PCMCodec struct {
	sampleRate int
	channels   int
	frameMs    int
}

// NewPCMCodec creates a passthrough codec with 20ms frames.
func NewPCMCodec(sampleRate, channels int) *PCMCodec {
	return &PCMCodec{sampleRate: sampleRate, channels: channels, frameMs: 20}
}

func (c *PCMCodec) Type() CodecType                    { return PCM }
func (c *PCMCodec) ID() uint8                          { return 0 }
func (c *PCMCodec) Encode(pcm []byte) ([]byte, error)  { return pcm, nil }
func (c *PCMCodec) Decode(data []byte) ([]byte, error) { return data, nil }
func (c *PCMCodec) FrameDurationMs() int               { return c.frameMs }
func (c *PCMCodec) SampleRate() int                    { return c.sampleRate }
func (c *PCMCodec) Channels() int                      { return c.channels }
func (c *PCMCodec) PCMFrameBytes() int {
	return c.sampleRate * c.channels * 2 * c.frameMs / 1000
}
func (c *PCMCodec) EncodedFrameBytes() int { return c.PCMFrameBytes() }
