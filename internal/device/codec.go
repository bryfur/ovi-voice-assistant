package device

import (
	"fmt"
	"log/slog"
)

// CodecType identifies a codec.
type CodecType string

const (
	PCM  CodecType = "pcm"
	LC3  CodecType = "lc3"
	Opus CodecType = "opus"
)

// parseCodecType validates a codec name.
func parseCodecType(s string) (CodecType, error) {
	switch CodecType(s) {
	case PCM, LC3, Opus:
		return CodecType(s), nil
	}
	return "", fmt.Errorf("unknown codec: %q", s)
}

// AudioCodec encodes and decodes audio frames.
type AudioCodec interface {
	// Type returns the codec type.
	Type() CodecType
	// ID returns the wire protocol codec id: 0=PCM, 1=LC3, 2=Opus.
	ID() uint8
	// Encode encodes one frame of PCM to compressed bytes.
	Encode(pcm []byte) ([]byte, error)
	// Decode decodes one compressed frame to PCM bytes.
	Decode(data []byte) ([]byte, error)
	// FrameDurationMs is the duration of one codec frame in milliseconds.
	FrameDurationMs() int
	// PCMFrameBytes is the PCM bytes per codec frame
	// (sample_rate * channels * 2 * frame_duration_ms / 1000).
	PCMFrameBytes() int
	// SampleRate is the actual sample rate used by the codec (may differ
	// from the requested rate).
	SampleRate() int
	// Channels is the number of audio channels (1=mono, 2=stereo).
	Channels() int
	// EncodedFrameBytes is the encoded bytes per codec frame, per channel.
	//
	// For multi-channel compressed codecs (e.g. LC3 stereo), this is the
	// per-channel encoded byte count — the actual wire frame size is
	// EncodedFrameBytes * Channels. This matches the ESP32
	// esp_lc3_dec_cfg_t.nbyte field so AUDIO_CONFIG can be forwarded to
	// the device decoder verbatim.
	EncodedFrameBytes() int
}

// Valid sample rates per codec.
var (
	lc3ValidRates  = []int{8000, 16000, 24000, 32000, 48000}
	opusValidRates = []int{8000, 12000, 16000, 24000, 48000}
)

// nearestValidRate returns the closest supported sample rate. Ties pick the
// lower rate.
func nearestValidRate(rate int, valid []int) int {
	best := valid[0]
	bestDiff := absInt(rate - best)
	for _, r := range valid[1:] {
		d := absInt(rate - r)
		if d < bestDiff {
			best, bestDiff = r, d
		}
	}
	return best
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// newCodec builds a codec instance. nbyte is the encoded bytes per channel
// per 10 ms (bitrate = nbyte × 800 bps per channel); 0 selects the voice
// default. LC3 uses it as its frame size; Opus derives its bitrate from it
// and switches to audio mode. PCM ignores it.
func newCodec(codecType CodecType, sampleRate, channels, nbyte int) (AudioCodec, error) {
	if channels <= 0 {
		channels = 1
	}
	switch codecType {
	case PCM:
		return NewPCMCodec(sampleRate, channels), nil
	case LC3:
		valid := nearestValidRate(sampleRate, lc3ValidRates)
		if valid != sampleRate {
			slog.Warn("LC3 does not support requested rate, using nearest valid rate",
				"requested", sampleRate, "using", valid)
		}
		if nbyte <= 0 {
			nbyte = lc3DefaultNByte
		}
		return newLC3Codec(valid, channels, nbyte)
	case Opus:
		valid := nearestValidRate(sampleRate, opusValidRates)
		if valid != sampleRate {
			slog.Warn("Opus does not support requested rate, using nearest valid rate",
				"requested", sampleRate, "using", valid)
		}
		return newOpusCodec(valid, channels, nbyte)
	}
	return nil, fmt.Errorf("unknown codec: %q", codecType)
}

// NewCodec builds a codec by name; see newCodec for nbyte.
func NewCodec(name string, sampleRate, channels, nbyte int) (AudioCodec, error) {
	ct, err := parseCodecType(name)
	if err != nil {
		return nil, err
	}
	return newCodec(ct, sampleRate, channels, nbyte)
}

// Describe renders a codec's audio settings for logs, e.g.
// "lc3 48000Hz 2ch 10ms 60B/ch 48kbps/ch".
func Describe(c AudioCodec) string {
	if c.Type() == PCM {
		return fmt.Sprintf("pcm %dHz %dch 16-bit %dms", c.SampleRate(), c.Channels(), c.FrameDurationMs())
	}
	kbps := c.EncodedFrameBytes() * 8 / c.FrameDurationMs() // per channel
	return fmt.Sprintf("%s %dHz %dch %dms %dB/ch %dkbps/ch",
		c.Type(), c.SampleRate(), c.Channels(), c.FrameDurationMs(), c.EncodedFrameBytes(), kbps)
}

// NameForID maps a wire codec id to its name.
func NameForID(id uint8) CodecType {
	switch id {
	case 1:
		return LC3
	case 2:
		return Opus
	}
	return PCM
}
