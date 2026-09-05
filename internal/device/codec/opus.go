package codec

import (
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/tphakala/go-opus/opus"
)

const (
	opusFrameMs   = 20
	opusVoiceMax  = 80   // typical ceiling of a 20 ms voice packet, informational
	opusPacketMax = 1276 // holds any single-frame packet
)

// opusCodec is a pure Go Opus codec with 20 ms frames. nbyte = 0 leaves the
// bitrate to the encoder; otherwise it targets nbyte × 800 bps per channel.
type opusCodec struct {
	f       Format
	bitrate int
	mu      sync.Mutex
	enc     *opus.Encoder
	dec     *opus.Decoder
}

func newOpus(rate, channels, nbyte int) (*opusCodec, error) {
	cfg := opus.EncoderConfig{SampleRate: rate, Channels: channels, ConstrainedVBR: true}
	frameBytes := opusVoiceMax
	if nbyte > 0 {
		cfg.Bitrate, frameBytes = nbyte*800*channels, nbyte*opusFrameMs/10
	}
	enc, err := opus.NewEncoder(cfg)
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	dec, err := opus.NewDecoder(rate, channels)
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	f := Format{Type: Opus, Rate: rate, Channels: channels, FrameMs: opusFrameMs, FrameBytes: frameBytes}
	return &opusCodec{f: f, bitrate: cfg.Bitrate, enc: enc, dec: dec}, nil
}

func (c *opusCodec) Format() Format { return c.f }

func (c *opusCodec) Encode(pcm []byte) ([]byte, error) {
	need := c.f.PCMBytes()
	if len(pcm) < need {
		return nil, fmt.Errorf("opus: need %d PCM bytes, got %d", need, len(pcm))
	}
	samples := make([]int16, need/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
	}
	out := make([]byte, opusPacketMax)
	c.mu.Lock()
	n, err := c.enc.Encode(samples, out)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	return out[:n], nil
}

// Decode expands a packet; an empty one is concealed as a lost frame.
func (c *opusCodec) Decode(frame []byte) ([]byte, error) {
	samples := make([]int16, c.f.PCMBytes()/2)
	c.mu.Lock()
	n, err := c.dec.Decode(frame, samples)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	pcm := make([]byte, 2*n*c.f.Channels)
	for i := range n * c.f.Channels {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(samples[i]))
	}
	return pcm, nil
}
