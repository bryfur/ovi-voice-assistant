"""PipelineOutput — interface for sending pipeline events and PCM audio to a device."""

from abc import ABC, abstractmethod

from ovi_voice_assistant.transport import EventType


class PipelineOutput(ABC):
    """Interface for sending pipeline events and PCM audio to a device."""

    @abstractmethod
    async def send_event(self, event: EventType, payload: bytes = b"") -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...
