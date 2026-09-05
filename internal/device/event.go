// Package device talks to ESPHome voice devices: control events and audio
// frames over WiFi (TCP) or BLE (GATT), and a paced speaker that feeds a
// device's decoder in real time.
package device

import (
	"encoding/binary"
	"fmt"

	"github.com/bryfur/ovi-voice-assistant/internal/device/codec"
)

// Event is a control message. The value is the wire byte.
type Event uint8

const (
	EventWakeWord    Event = iota + 1 // device → server: wake word heard; payload [2B peak][2B ambient][word]
	EventVADStart                     // server → device: the user started talking
	EventMicStop                      // server → device: stop streaming the mic
	EventTTSStart                     // server → device: speaker audio follows
	EventTTSEnd                       // server → device: speaker audio done
	EventContinue                     // server → device: keep listening after this reply
	EventError                        // server → device: "code\x00message"
	EventAudioConfig                  // server → device: speaker format, see AudioConfig
	EventMicConfig                    // both ways: mic format, see MicConfig
	EventWakeAbort                    // server → device: another device won the wake word
	EventSyncPlay                     // server → device: start playback at this NTP time (8B LE ms)
)

var eventNames = [...]string{"", "WAKE_WORD", "VAD_START", "MIC_STOP", "TTS_START", "TTS_END",
	"CONTINUE", "ERROR", "AUDIO_CONFIG", "MIC_CONFIG", "WAKE_ABORT", "SYNC_PLAY"}

func (e Event) String() string {
	if e.valid() {
		return eventNames[e]
	}
	return fmt.Sprintf("EVENT_0x%02X", uint8(e))
}

func (e Event) valid() bool { return e > 0 && int(e) < len(eventNames) }

// Audio frame message types.
const (
	micAudio     = 0x20 // device → server
	speakerAudio = 0x21 // server → device
)

// AudioConfig is the AUDIO_CONFIG payload: [4B rate][2B frame bytes][1B codec][1B channels].
func AudioConfig(f codec.Format) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint32(b, uint32(f.Rate))
	binary.LittleEndian.PutUint16(b[4:], uint16(f.FrameBytes))
	b[6], b[7] = f.Type.ID(), uint8(f.Channels)
	return b
}

// MicConfig is the MIC_CONFIG payload: [4B rate][2B frame bytes][1B codec].
func MicConfig(f codec.Format) []byte { return AudioConfig(f)[:7] }

// ParseFormat reads either payload back into a Format (FrameMs is unknown
// on the wire and left zero).
func ParseFormat(b []byte) (codec.Format, error) {
	if len(b) < 7 {
		return codec.Format{}, fmt.Errorf("format payload too short: %d bytes", len(b))
	}
	f := codec.Format{
		Type:       codec.TypeOf(b[6]),
		Rate:       int(binary.LittleEndian.Uint32(b)),
		FrameBytes: int(binary.LittleEndian.Uint16(b[4:])),
		Channels:   1,
	}
	if len(b) >= 8 {
		f.Channels = int(b[7])
	}
	return f, nil
}

// ParseWake reads a WAKE_WORD payload: [2B peak][2B ambient][word], or
// just the word from older firmware. The score is peak/ambient x1000, so
// the device nearest the speaker wins regardless of mic gain.
func ParseWake(b []byte) (word string, score int) {
	if len(b) < 4 {
		return string(b), 0
	}
	peak, ambient := int(binary.LittleEndian.Uint16(b)), int(binary.LittleEndian.Uint16(b[2:]))
	return string(b[4:]), peak * 1000 / max(ambient, 1)
}
