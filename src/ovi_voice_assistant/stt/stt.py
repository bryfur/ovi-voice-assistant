"""Speech-to-text base class."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable

from ovi_voice_assistant.config import Settings

# Callback fired when VAD detects speech start
VadStartCallback = Callable[[], Awaitable[None]]


class STT(ABC):
    @abstractmethod
    def __init__(self, settings: Settings) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str: ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        on_vad_start: VadStartCallback | None = None,
    ) -> str: ...
