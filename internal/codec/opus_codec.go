package codec

import (
	"encoding/binary"
	"fmt"
	"sync"

	"gopkg.in/hraban/opus.v2"
)

const (
	OpusFrameDurationMs  = 20
	OpusMaxEncodedBytes  = 80   // typical max for a 20 ms VoIP frame (informational)
	opusEncodeBufferSize = 4000 // upper bound for a single packet
)

// OpusCodec is the Opus audio codec (20ms frames, VBR).
type OpusCodec struct {
	sampleRate   int
	channels     int
	frameSamples int
	frameBytes   int // per channel per frame, informational

	mu  sync.Mutex
	enc *opus.Encoder
	dec *opus.Decoder
}

// NewOpusCodec creates an Opus codec. nbyte = 0 is VoIP mode at the
// library's default bitrate; otherwise audio mode at nbyte × 800 bps per
// channel (nbyte is bytes per channel per 10 ms).
func NewOpusCodec(sampleRate, channels, nbyte int) (*OpusCodec, error) {
	if channels <= 0 {
		channels = 1
	}
	app, frameBytes := opus.AppVoIP, OpusMaxEncodedBytes
	if nbyte > 0 {
		app, frameBytes = opus.AppAudio, nbyte*OpusFrameDurationMs/10
	}
	enc, err := opus.NewEncoder(sampleRate, channels, app)
	if err != nil {
		return nil, fmt.Errorf("opus encoder: %w", err)
	}
	if nbyte > 0 {
		if err := enc.SetBitrate(nbyte * 800 * channels); err != nil {
			return nil, fmt.Errorf("opus bitrate: %w", err)
		}
	}
	dec, err := opus.NewDecoder(sampleRate, channels)
	if err != nil {
		return nil, fmt.Errorf("opus decoder: %w", err)
	}
	return &OpusCodec{
		sampleRate:   sampleRate,
		channels:     channels,
		frameSamples: sampleRate * OpusFrameDurationMs / 1000,
		frameBytes:   frameBytes,
		enc:          enc,
		dec:          dec,
	}, nil
}

// Bitrate returns the encoder's configured bitrate in bps.
func (c *OpusCodec) Bitrate() int {
	b, _ := c.enc.Bitrate()
	return b
}

func (c *OpusCodec) Type() CodecType        { return Opus }
func (c *OpusCodec) ID() uint8              { return 2 }
func (c *OpusCodec) SampleRate() int        { return c.sampleRate }
func (c *OpusCodec) Channels() int          { return c.channels }
func (c *OpusCodec) FrameDurationMs() int   { return OpusFrameDurationMs }
func (c *OpusCodec) PCMFrameBytes() int     { return c.frameSamples * c.channels * 2 }
func (c *OpusCodec) EncodedFrameBytes() int { return c.frameBytes } // VBR; informational

// Encode encodes one 20ms PCM frame.
func (c *OpusCodec) Encode(pcm []byte) ([]byte, error) {
	need := c.PCMFrameBytes()
	if len(pcm) < need {
		return nil, fmt.Errorf("opus: need %d PCM bytes, got %d", need, len(pcm))
	}
	samples := bytesToInt16(pcm[:need])
	buf := make([]byte, opusEncodeBufferSize)
	c.mu.Lock()
	n, err := c.enc.Encode(samples, buf)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("opus encode: %w", err)
	}
	return buf[:n], nil
}

// Decode decodes one Opus packet to a 20ms PCM frame.
func (c *OpusCodec) Decode(data []byte) ([]byte, error) {
	samples := make([]int16, c.frameSamples*c.channels)
	c.mu.Lock()
	n, err := c.dec.Decode(data, samples)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("opus decode: %w", err)
	}
	return int16ToBytes(samples[:n*c.channels]), nil
}

func bytesToInt16(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

func int16ToBytes(s []int16) []byte {
	out := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}
	return out
}
