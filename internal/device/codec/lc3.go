package codec

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/caitunai/lc3"
)

const lc3FrameUs = 10_000

// lc3Codec runs one liblc3 encoder/decoder pair per channel; the wire
// frame is the per-channel blocks concatenated. liblc3 is compiled into
// the binary from the sources bundled with the wrapper module.
type lc3Codec struct {
	f       Format
	mu      sync.Mutex
	enc     []*lc3.Encoder
	dec     []*lc3.Decoder
	samples []float32 // one channel of one frame
}

func newLC3(rate, channels, nbyte int) (*lc3Codec, error) {
	cfg := lc3.Config{SampleRate: rate, FrameDurationUS: lc3FrameUs, FrameBytes: nbyte}
	c := &lc3Codec{f: Format{Type: LC3, Rate: rate, Channels: channels, FrameMs: lc3FrameUs / 1000, FrameBytes: nbyte}}
	for range channels {
		enc, err := lc3.NewEncoder(cfg)
		if err == nil {
			var dec *lc3.Decoder
			if dec, err = lc3.NewDecoder(cfg); err == nil {
				c.enc, c.dec = append(c.enc, enc), append(c.dec, dec)
				continue
			}
		}
		c.Close()
		return nil, fmt.Errorf("lc3: %d Hz, %d bytes per frame: %w", rate, nbyte, err)
	}
	c.samples = make([]float32, c.enc[0].FrameSize())
	runtime.SetFinalizer(c, (*lc3Codec).Close)
	return c, nil
}

func (c *lc3Codec) Format() Format { return c.f }

// Close frees the native state; the codec is unusable afterwards.
func (c *lc3Codec) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.enc {
		_ = e.Close()
	}
	for _, d := range c.dec {
		_ = d.Close()
	}
	c.enc, c.dec = nil, nil
}

func (c *lc3Codec) Encode(pcm []byte) ([]byte, error) {
	if len(pcm) < c.f.PCMBytes() {
		return nil, fmt.Errorf("lc3: need %d PCM bytes, got %d", c.f.PCMBytes(), len(pcm))
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.enc == nil {
		return nil, fmt.Errorf("lc3: closed")
	}
	n, stride := c.f.FrameBytes, c.f.Channels
	out := make([]byte, n*stride)
	for ch, enc := range c.enc {
		for i := range c.samples {
			c.samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*(i*stride+ch):]))) / 32768
		}
		frame, err := enc.Encode(c.samples)
		if err != nil {
			return nil, fmt.Errorf("lc3: %w", err)
		}
		copy(out[ch*n:], frame)
	}
	return out, nil
}

// Decode expands a frame; an empty one decodes as silence.
func (c *lc3Codec) Decode(frame []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dec == nil {
		return nil, fmt.Errorf("lc3: closed")
	}
	pcm := make([]byte, c.f.PCMBytes())
	n, stride := len(frame)/c.f.Channels, c.f.Channels
	for ch, dec := range c.dec {
		samples, err := dec.Decode(frame[ch*n : (ch+1)*n])
		if err != nil {
			return nil, fmt.Errorf("lc3: %w", err)
		}
		for i, s := range samples {
			binary.LittleEndian.PutUint16(pcm[2*(i*stride+ch):], uint16(sample(s)))
		}
	}
	return pcm, nil
}

// sample rounds a [-1, 1] value to int16 the way liblc3 itself does.
func sample(v float32) int16 {
	return int16(max(-32768, min(32767, math.Round(float64(v)*32768))))
}
