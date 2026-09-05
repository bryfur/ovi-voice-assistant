// Package transport defines the device transport abstraction for Ovi —
// WiFi (plain TCP) and BLE (GATT).
package transport

import (
	"encoding/binary"
	"fmt"
)

// EventType is a control event shared across all transports.
type EventType uint8

const (
	// Device → Server
	EventWakeWord EventType = 0x01 // device→server: wake word detected
	// Server → Device
	EventVADStart EventType = 0x02 // server→device: speech detected (user started talking)
	EventMicStop  EventType = 0x03 // server→device: stop recording mic
	EventTTSStart EventType = 0x04 // server→device: speaker audio starting
	EventTTSEnd   EventType = 0x05 // server→device: speaker audio done
	EventContinue EventType = 0x06 // server→device: keep listening (follow-up)
	EventError    EventType = 0x07 // server→device: error (code\0message)
	// Bidirectional
	EventAudioConfig EventType = 0x08 // bidirectional: speaker codec config
	EventMicConfig   EventType = 0x09 // bidirectional: mic codec config
	EventWakeAbort   EventType = 0x0A // server→device: abort wake (another device won)
	EventSyncPlay    EventType = 0x0B // server→device: start playback at NTP timestamp (8B LE ms)
)

var eventNames = map[EventType]string{
	EventWakeWord:    "WAKE_WORD",
	EventVADStart:    "VAD_START",
	EventMicStop:     "MIC_STOP",
	EventTTSStart:    "TTS_START",
	EventTTSEnd:      "TTS_END",
	EventContinue:    "CONTINUE",
	EventError:       "ERROR",
	EventAudioConfig: "AUDIO_CONFIG",
	EventMicConfig:   "MIC_CONFIG",
	EventWakeAbort:   "WAKE_ABORT",
	EventSyncPlay:    "SYNC_PLAY",
}

// String returns the symbolic event name.
func (e EventType) String() string {
	if name, ok := eventNames[e]; ok {
		return name
	}
	return fmt.Sprintf("EVENT_0x%02X", uint8(e))
}

// Valid reports whether the byte is a known control event.
func (e EventType) Valid() bool {
	_, ok := eventNames[e]
	return ok
}

// AudioConfig is the payload of an AUDIO_CONFIG event.
type AudioConfig struct {
	SampleRate        uint32
	EncodedFrameBytes uint16 // 0 = PCM
	CodecType         uint8  // 0=PCM, 1=LC3, 2=Opus
	Channels          uint8  // 1=mono, 2=stereo
}

// Marshal packs the config as <IHBB little-endian.
func (c AudioConfig) Marshal() []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint32(buf[0:], c.SampleRate)
	binary.LittleEndian.PutUint16(buf[4:], c.EncodedFrameBytes)
	buf[6] = c.CodecType
	buf[7] = c.Channels
	return buf
}

// UnmarshalAudioConfig parses a <IHBB payload.
func UnmarshalAudioConfig(payload []byte) (AudioConfig, error) {
	if len(payload) < 8 {
		return AudioConfig{}, fmt.Errorf("audio config payload too short: %d bytes", len(payload))
	}
	return AudioConfig{
		SampleRate:        binary.LittleEndian.Uint32(payload[0:]),
		EncodedFrameBytes: binary.LittleEndian.Uint16(payload[4:]),
		CodecType:         payload[6],
		Channels:          payload[7],
	}, nil
}

// Callback types.
type (
	EventCallback      func(event EventType, payload []byte)
	AudioCallback      func(data []byte)
	DisconnectCallback func()
	ConnectCallback    func()
)

// DeviceTransport is the abstract transport for communicating with a voice
// device. Callbacks are invoked from the transport's receive goroutine and
// must not block for long.
type DeviceTransport interface {
	// Connect establishes the connection to the device.
	Connect() error
	// Disconnect closes the connection and stops reconnection attempts.
	Disconnect() error
	// SendEvent sends a control event to the device.
	SendEvent(event EventType, payload []byte) error
	// SendAudio sends encoded audio to the device speaker.
	SendAudio(data []byte) error
	// SetEventCallback registers the callback for device-to-server events.
	SetEventCallback(cb EventCallback)
	// SetAudioCallback registers the callback for device-to-server audio.
	SetAudioCallback(cb AudioCallback)
	// SetDisconnectCallback registers the callback for disconnection.
	SetDisconnectCallback(cb DisconnectCallback)
	// SetConnectCallback registers the callback for (re)connection.
	SetConnectCallback(cb ConnectCallback)
	// IsConnected reports whether the transport is currently connected.
	IsConnected() bool
	// String describes the transport for logging.
	String() string
}

// SendAudioConfig sends an AUDIO_CONFIG event with codec parameters.
func SendAudioConfig(t DeviceTransport, cfg AudioConfig) error {
	return t.SendEvent(EventAudioConfig, cfg.Marshal())
}
