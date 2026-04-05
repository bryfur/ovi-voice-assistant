from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from open_voice_assistant.config import Settings


class STT(ABC):
    @abstractmethod
    def __init__(self, settings: Settings) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str: ...

    @abstractmethod
    async def transcribe_stream(self, audio_chunks: AsyncIterator[bytes]) -> str: ...
