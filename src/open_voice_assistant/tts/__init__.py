from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import numpy as np

from open_voice_assistant.config import Settings


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 16-bit PCM audio using linear interpolation."""
    ratio = dst_rate / src_rate
    new_length = int(len(audio) * ratio)
    indices = np.arange(new_length) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    left = np.floor(indices).astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = indices - left
    return (audio[left] * (1 - frac) + audio[right] * frac).astype(np.int16)


class TTS(ABC):
    sample_rate: int    # Hz, set after load()
    sample_width: int   # bytes per sample (2 = 16-bit)
    channels: int       # 1 = mono, 2 = stereo

    @abstractmethod
    def __init__(self, settings: Settings) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    @abstractmethod
    def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]: ...
