"""Transport abstraction for Ovi — WiFi (native API) and BLE."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum


class EventType(IntEnum):
    """Control events — shared across all transports."""

    # Device → Server
    WAKE_WORD = 0x01  # device→server: wake word detected
    # Server → Device
    VAD_START = 0x02  # server→device: speech detected (user started talking)
    MIC_STOP = 0x03  # server→device: stop recording mic
    TTS_START = 0x04  # server→device: speaker audio starting
    TTS_END = 0x05  # server→device: speaker audio done
    CONTINUE = 0x06  # server→device: keep listening (follow-up)
    ERROR = 0x07  # server→device: error (code\0message)
    # Bidirectional
    AUDIO_CONFIG = 0x08  # bidirectional: speaker codec config
    MIC_CONFIG = 0x09  # bidirectional: mic codec config
    WAKE_ABORT = 0x0A  # server→device: abort wake (another device won)
    SYNC_PLAY = 0x0B  # server→device: start playback at NTP timestamp (8B LE ms)


@dataclass
class AudioConfig:
    """Audio configuration sent via AUDIO_CONFIG event."""

    sample_rate: int
    encoded_frame_bytes: int  # 0 = PCM
    codec_type: int  # 0=PCM, 1=LC3, 2=Opus
    channels: int = 1  # 1=mono, 2=stereo


# Callback types
EventCallback = Callable[[EventType, bytes], Awaitable[None]]
AudioCallback = Callable[[bytes], Awaitable[None]]
DisconnectCallback = Callable[[], Awaitable[None]]
ConnectCallback = Callable[[], Awaitable[None]]


class DeviceTransport(ABC):
    """Abstract transport for communicating with a voice device."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the device."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the device."""
        ...

    @abstractmethod
    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        """Send a control event to the device."""
        ...

    @abstractmethod
    async def send_audio(self, data: bytes) -> None:
        """Send encoded audio to the device speaker."""
        ...

    @abstractmethod
    def set_event_callback(self, callback: EventCallback) -> None:
        """Register callback for device-to-server events."""
        ...

    @abstractmethod
    def set_audio_callback(self, callback: AudioCallback) -> None:
        """Register callback for device-to-server audio."""
        ...

    @abstractmethod
    def set_disconnect_callback(self, callback: DisconnectCallback) -> None:
        """Register callback for disconnection."""
        ...

    @abstractmethod
    def set_connect_callback(self, callback: ConnectCallback) -> None:
        """Register callback for (re)connection."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    async def send_audio_config(self, config: AudioConfig) -> None:
        """Send AUDIO_CONFIG event with codec parameters."""
        import struct

        payload = struct.pack(
            "<IHBB",
            config.sample_rate,
            config.encoded_frame_bytes,
            config.codec_type,
            config.channels,
        )
        await self.send_event(EventType.AUDIO_CONFIG, payload)
