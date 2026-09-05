package codec

/*
#cgo pkg-config: lc3
#include <stdlib.h>
#include <lc3.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

const lc3FrameUs = 10_000

// lc3 runs one liblc3 encoder/decoder pair per channel; the wire frame is
// the per-channel blocks concatenated.
type lc3 struct {
	f        Format
	mu       sync.Mutex
	channels []lc3Channel
}

type lc3Channel struct {
	enc C.lc3_encoder_t
	dec C.lc3_decoder_t
	mem [2]unsafe.Pointer
}

func newLC3(rate, channels, nbyte int) (*lc3, error) {
	samples := int(C.lc3_frame_samples(lc3FrameUs, C.int(rate)))
	if samples <= 0 {
		return nil, fmt.Errorf("lc3: unsupported rate %d", rate)
	}
	c := &lc3{f: Format{Type: LC3, Rate: rate, Channels: channels, FrameMs: lc3FrameUs / 1000, FrameBytes: nbyte}}
	for range channels {
		ch := lc3Channel{mem: [2]unsafe.Pointer{
			C.malloc(C.size_t(C.lc3_encoder_size(lc3FrameUs, C.int(rate)))),
			C.malloc(C.size_t(C.lc3_decoder_size(lc3FrameUs, C.int(rate)))),
		}}
		ch.enc = C.lc3_setup_encoder(lc3FrameUs, C.int(rate), C.int(rate), ch.mem[0])
		ch.dec = C.lc3_setup_decoder(lc3FrameUs, C.int(rate), C.int(rate), ch.mem[1])
		c.channels = append(c.channels, ch)
		if ch.enc == nil || ch.dec == nil {
			c.Close()
			return nil, fmt.Errorf("lc3: cannot set up codec at %d Hz", rate)
		}
	}
	runtime.SetFinalizer(c, (*lc3).Close)
	return c, nil
}

func (c *lc3) Format() Format { return c.f }

// Close frees the native state; the codec is unusable afterwards.
func (c *lc3) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, ch := range c.channels {
		C.free(ch.mem[0])
		C.free(ch.mem[1])
	}
	c.channels = nil
}

func (c *lc3) Encode(pcm []byte) ([]byte, error) {
	if len(pcm) < c.f.PCMBytes() {
		return nil, fmt.Errorf("lc3: need %d PCM bytes, got %d", c.f.PCMBytes(), len(pcm))
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.channels == nil {
		return nil, fmt.Errorf("lc3: closed")
	}
	n := c.f.FrameBytes
	out := make([]byte, n*len(c.channels))
	for i, ch := range c.channels {
		if rc := C.lc3_encode(ch.enc, C.LC3_PCM_FORMAT_S16, unsafe.Pointer(&pcm[i*2]), C.int(len(c.channels)),
			C.int(n), unsafe.Pointer(&out[i*n])); rc != 0 {
			return nil, fmt.Errorf("lc3: encode failed (%d)", rc)
		}
	}
	return out, nil
}

func (c *lc3) Decode(frame []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.channels == nil {
		return nil, fmt.Errorf("lc3: closed")
	}
	pcm := make([]byte, c.f.PCMBytes())
	n := len(frame) / len(c.channels)
	for i, ch := range c.channels {
		var in unsafe.Pointer // nil asks liblc3 to conceal a lost frame
		if n > 0 {
			in = unsafe.Pointer(&frame[i*n])
		}
		if rc := C.lc3_decode(ch.dec, in, C.int(n), C.LC3_PCM_FORMAT_S16, unsafe.Pointer(&pcm[i*2]),
			C.int(len(c.channels))); rc < 0 {
			return nil, fmt.Errorf("lc3: decode failed (%d)", rc)
		}
	}
	return pcm, nil
}
