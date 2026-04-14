"""Audio codec base class and codec type enum."""

from abc import ABC, abstractmethod
from enum import StrEnum


class CodecType(StrEnum):
    PCM = "pcm"
    LC3 = "lc3"
    OPUS = "opus"


class AudioCodec(ABC):
    """Encode/decode audio frames."""

    codec_type: CodecType  # set by subclasses
    codec_id: int  # 0=PCM, 1=LC3, 2=Opus — for wire protocol

    @abstractmethod
    def __init__(self, sample_rate: int, channels: int = 1) -> None: ...

    @abstractmethod
    def encode(self, pcm: bytes) -> bytes:
        """Encode one frame of PCM to compressed bytes."""
        ...

    @abstractmethod
    def decode(self, data: bytes) -> bytes:
        """Decode one compressed frame to PCM bytes."""
        ...

    @property
    @abstractmethod
    def frame_duration_ms(self) -> int:
        """Duration of one codec frame in milliseconds."""
        ...

    @property
    @abstractmethod
    def pcm_frame_bytes(self) -> int:
        """PCM bytes per codec frame (sample_rate * channels * 2 * frame_duration_ms / 1000)."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Actual sample rate used by the codec (may differ from requested rate)."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        ...

    @property
    @abstractmethod
    def encoded_frame_bytes(self) -> int:
        """Encoded bytes per codec frame, per channel.

        For multi-channel compressed codecs (e.g. LC3 stereo), this is the
        per-channel encoded byte count — the actual wire frame size is
        ``encoded_frame_bytes * channels``. This matches the ESP32
        ``esp_lc3_dec_cfg_t.nbyte`` field so AUDIO_CONFIG can be forwarded
        to the device decoder verbatim.
        """
        ...
