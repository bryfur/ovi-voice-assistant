package device

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

// Encoded bytes per channel per 10 ms frame (bitrate = nbyte × 800 bps).
const (
	lc3FrameDurationUs = 10_000
	lc3DefaultNByte    = 40 // 32 kbps per channel: voice
	LC3MusicNByte      = 80 // 64 kbps per channel: music, 48 kHz stereo
)

// lc3Codec is the LC3 audio codec (10ms frames).
//
// nbyte is the per-channel encoded byte count — matching the ESP32
// esp_lc3_dec_cfg_t.nbyte definition. The wire frame size is
// nbyte * channels (channels are concatenated in the encoded frame).
type lc3Codec struct {
	sampleRate   int
	channels     int
	nbyte        int
	frameSamples int

	mu       sync.Mutex
	encMem   []unsafe.Pointer
	decMem   []unsafe.Pointer
	encoders []C.lc3_encoder_t
	decoders []C.lc3_decoder_t
}

// newLC3Codec creates an LC3 codec with one encoder/decoder per channel.
func newLC3Codec(sampleRate, channels, nbyte int) (*lc3Codec, error) {
	if channels <= 0 {
		channels = 1
	}
	if nbyte <= 0 {
		nbyte = lc3DefaultNByte
	}
	frameSamples := int(C.lc3_frame_samples(C.int(lc3FrameDurationUs), C.int(sampleRate)))
	if frameSamples <= 0 {
		return nil, fmt.Errorf("lc3: unsupported sample rate %d", sampleRate)
	}
	c := &lc3Codec{
		sampleRate:   sampleRate,
		channels:     channels,
		nbyte:        nbyte,
		frameSamples: frameSamples,
	}
	encSize := C.lc3_encoder_size(C.int(lc3FrameDurationUs), C.int(sampleRate))
	decSize := C.lc3_decoder_size(C.int(lc3FrameDurationUs), C.int(sampleRate))
	for i := 0; i < channels; i++ {
		em := C.malloc(C.size_t(encSize))
		enc := C.lc3_setup_encoder(C.int(lc3FrameDurationUs), C.int(sampleRate), C.int(sampleRate), em)
		dm := C.malloc(C.size_t(decSize))
		dec := C.lc3_setup_decoder(C.int(lc3FrameDurationUs), C.int(sampleRate), C.int(sampleRate), dm)
		if enc == nil || dec == nil {
			C.free(em)
			C.free(dm)
			c.Close()
			return nil, fmt.Errorf("lc3: failed to set up codec at %d Hz", sampleRate)
		}
		c.encMem = append(c.encMem, em)
		c.decMem = append(c.decMem, dm)
		c.encoders = append(c.encoders, enc)
		c.decoders = append(c.decoders, dec)
	}
	runtime.SetFinalizer(c, func(c *lc3Codec) { c.Close() })
	return c, nil
}

// Close releases the native codec state.
func (c *lc3Codec) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, m := range c.encMem {
		C.free(m)
	}
	for _, m := range c.decMem {
		C.free(m)
	}
	c.encMem, c.decMem, c.encoders, c.decoders = nil, nil, nil, nil
}

func (c *lc3Codec) Type() CodecType        { return LC3 }
func (c *lc3Codec) ID() uint8              { return 1 }
func (c *lc3Codec) SampleRate() int        { return c.sampleRate }
func (c *lc3Codec) Channels() int          { return c.channels }
func (c *lc3Codec) FrameDurationMs() int   { return lc3FrameDurationUs / 1000 }
func (c *lc3Codec) PCMFrameBytes() int     { return c.frameSamples * c.channels * 2 }
func (c *lc3Codec) EncodedFrameBytes() int { return c.nbyte }

// Encode encodes one interleaved PCM frame. The output is the per-channel
// encoded blocks concatenated.
func (c *lc3Codec) Encode(pcm []byte) ([]byte, error) {
	need := c.PCMFrameBytes()
	if len(pcm) < need {
		return nil, fmt.Errorf("lc3: need %d PCM bytes, got %d", need, len(pcm))
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.encoders == nil {
		return nil, fmt.Errorf("lc3: codec closed")
	}
	out := make([]byte, c.nbyte*c.channels)
	for ch := 0; ch < c.channels; ch++ {
		rc := C.lc3_encode(
			c.encoders[ch], C.LC3_PCM_FORMAT_S16,
			unsafe.Pointer(&pcm[ch*2]), C.int(c.channels),
			C.int(c.nbyte), unsafe.Pointer(&out[ch*c.nbyte]),
		)
		if rc != 0 {
			return nil, fmt.Errorf("lc3: encode failed (%d)", int(rc))
		}
	}
	return out, nil
}

// Decode decodes one encoded frame to interleaved PCM.
func (c *lc3Codec) Decode(data []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.decoders == nil {
		return nil, fmt.Errorf("lc3: codec closed")
	}
	pcm := make([]byte, c.PCMFrameBytes())
	perCh := len(data) / c.channels
	for ch := 0; ch < c.channels; ch++ {
		var in unsafe.Pointer
		if perCh > 0 {
			in = unsafe.Pointer(&data[ch*perCh])
		}
		rc := C.lc3_decode(
			c.decoders[ch], in, C.int(perCh),
			C.LC3_PCM_FORMAT_S16, unsafe.Pointer(&pcm[ch*2]), C.int(c.channels),
		)
		if rc < 0 {
			return nil, fmt.Errorf("lc3: decode failed (%d)", int(rc))
		}
	}
	return pcm, nil
}
