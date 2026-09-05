// Package codec implements the frame codecs used on the wire: PCM, LC3 and Opus.
package codec

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
)

// Type names a codec. Its value is the name used in config and logs.
type Type string

const (
	PCM  Type = "pcm"
	LC3  Type = "lc3"
	Opus Type = "opus"
)

var ids = map[Type]uint8{PCM: 0, LC3: 1, Opus: 2}

// ID is the codec's wire protocol id.
func (t Type) ID() uint8 { return ids[t] }

// TypeOf maps a wire id back to a Type; unknown ids are PCM.
func TypeOf(id uint8) Type {
	for t, i := range ids {
		if i == id {
			return t
		}
	}
	return PCM
}

// Format describes one audio stream: what the frames contain and how big
// they are. FrameBytes is the encoded size per channel per frame; for
// multi-channel LC3 the wire frame is FrameBytes × Channels, which matches
// the ESP32 decoder's nbyte so the value can be forwarded verbatim.
type Format struct {
	Type       Type
	Rate       int // Hz
	Channels   int
	FrameMs    int // frame duration
	FrameBytes int // encoded bytes per channel per frame
}

// PCMBytes is the size of one frame of raw 16-bit PCM.
func (f Format) PCMBytes() int { return f.Rate * f.Channels * 2 * f.FrameMs / 1000 }

// Kbps is the per-channel bitrate.
func (f Format) Kbps() int { return f.FrameBytes * 8 / f.FrameMs }

// String renders the format for logs, e.g. "lc3 48000Hz 2ch 10ms 80B/ch 64kbps/ch".
func (f Format) String() string {
	if f.Type == PCM {
		return fmt.Sprintf("pcm %dHz %dch 16-bit %dms", f.Rate, f.Channels, f.FrameMs)
	}
	return fmt.Sprintf("%s %dHz %dch %dms %dB/ch %dkbps/ch", f.Type, f.Rate, f.Channels, f.FrameMs, f.FrameBytes, f.Kbps())
}

// Codec encodes and decodes one frame at a time.
type Codec interface {
	Format() Format
	// Encode compresses exactly one PCM frame (Format().PCMBytes() bytes).
	Encode(pcm []byte) ([]byte, error)
	// Decode expands one encoded frame to PCM.
	Decode(frame []byte) ([]byte, error)
}

// Encoded bytes per channel per 10 ms (bitrate = nbyte × 800 bps).
const (
	VoiceNByte = 40 // 32 kbps per channel
	MusicNByte = 80 // 64 kbps per channel at 48 kHz stereo
)

var (
	lc3Rates  = []int{8000, 16000, 24000, 32000, 48000}
	opusRates = []int{8000, 12000, 16000, 24000, 48000}
)

// New builds a codec by name. The rate snaps to the nearest one the codec
// supports. nbyte is the encoded bytes per channel per 10 ms; 0 selects the
// voice default. LC3 uses it as its frame size, Opus derives its bitrate
// from it and switches to audio mode, PCM ignores it.
func New(name string, rate, channels, nbyte int) (Codec, error) {
	channels = max(channels, 1)
	switch Type(name) {
	case PCM:
		return pcm{Format{Type: PCM, Rate: rate, Channels: channels, FrameMs: 20}}, nil
	case LC3:
		return newLC3(nearest(rate, lc3Rates), channels, cmp.Or(nbyte, VoiceNByte))
	case Opus:
		return newOpus(nearest(rate, opusRates), channels, nbyte)
	}
	return nil, fmt.Errorf("unknown codec %q", name)
}

// nearest returns the supported rate closest to want (ties favour the lower).
func nearest(want int, supported []int) int {
	got := slices.MinFunc(supported, func(a, b int) int {
		return cmp.Compare(abs(a-want), abs(b-want))
	})
	if got != want {
		slog.Warn("Codec does not support rate, using nearest", "requested", want, "using", got)
	}
	return got
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
