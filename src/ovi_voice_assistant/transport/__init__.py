"""Transport abstraction for Ovi — WiFi (native API) and BLE."""

from ovi_voice_assistant.transport.device_transport import (
    AudioCallback,
    AudioConfig,
    ConnectCallback,
    DeviceTransport,
    DisconnectCallback,
    EventCallback,
    EventType,
)

__all__ = [
    "AudioCallback",
    "AudioConfig",
    "ConnectCallback",
    "DeviceTransport",
    "DisconnectCallback",
    "EventCallback",
    "EventType",
]
