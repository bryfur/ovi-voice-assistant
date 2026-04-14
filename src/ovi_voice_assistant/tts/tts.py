"""Text-to-speech base class."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ovi_voice_assistant.config import Settings


class TTS(ABC):
    sample_rate: int  # Hz, set after load()
    sample_width: int  # bytes per sample (2 = 16-bit)
    channels: int  # 1 = mono, 2 = stereo

    @abstractmethod
    def __init__(self, settings: Settings, sample_rate: int = 16000) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    @abstractmethod
    def synthesize_stream(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]: ...
