package codec

import (
	"encoding/binary"
	"fmt"
	"sync"

	"gopkg.in/hraban/opus.v2"
)

const (
	opusFrameMs   = 20
	opusVoiceMax  = 80   // typical ceiling of a 20 ms VoIP packet, informational
	opusPacketMax = 4000 // encode buffer
)

// opusCodec wraps libopus with 20 ms frames. nbyte = 0 is VoIP mode at the
// library default bitrate; otherwise audio mode at nbyte × 800 bps per channel.
type opusCodec struct {
	f   Format
	mu  sync.Mutex
	enc *opus.Encoder
	dec *opus.Decoder
}

func newOpus(rate, channels, nbyte int) (*opusCodec, error) {
	app, frameBytes := opus.AppVoIP, opusVoiceMax
	if nbyte > 0 {
		app, frameBytes = opus.AppAudio, nbyte*opusFrameMs/10
	}
	enc, err := opus.NewEncoder(rate, channels, app)
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	if nbyte > 0 {
		if err := enc.SetBitrate(nbyte * 800 * channels); err != nil {
			return nil, fmt.Errorf("opus: %w", err)
		}
	}
	dec, err := opus.NewDecoder(rate, channels)
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	return &opusCodec{f: Format{Type: Opus, Rate: rate, Channels: channels, FrameMs: opusFrameMs, FrameBytes: frameBytes}, enc: enc, dec: dec}, nil
}

func (c *opusCodec) Format() Format { return c.f }

// Bitrate is the encoder's configured bitrate in bps.
func (c *opusCodec) Bitrate() int {
	b, _ := c.enc.Bitrate()
	return b
}

func (c *opusCodec) Encode(pcm []byte) ([]byte, error) {
	need := c.f.PCMBytes()
	if len(pcm) < need {
		return nil, fmt.Errorf("opus: need %d PCM bytes, got %d", need, len(pcm))
	}
	samples := make([]int16, need/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
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

func (c *opusCodec) Decode(frame []byte) ([]byte, error) {
	samples := make([]int16, c.f.PCMBytes()/2)
	c.mu.Lock()
	n, err := c.dec.Decode(frame, samples)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("opus: %w", err)
	}
	pcm := make([]byte, n*c.f.Channels*2)
	for i := range n * c.f.Channels {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(samples[i]))
	}
	return pcm, nil
}
