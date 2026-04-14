"""PCM passthrough codec -- no encoding/decoding."""

from ovi_voice_assistant.codec.audio_codec import AudioCodec, CodecType


class PcmCodec(AudioCodec):
    """Identity codec that passes PCM audio through unchanged."""

    codec_type = CodecType.PCM
    codec_id = 0

    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_ms = 20  # 20ms frames

    def encode(self, pcm: bytes) -> bytes:
        return pcm

    def decode(self, data: bytes) -> bytes:
        return data

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        return self._frame_ms

    @property
    def pcm_frame_bytes(self) -> int:
        return self._sample_rate * self._channels * 2 * self._frame_ms // 1000

    @property
    def encoded_frame_bytes(self) -> int:
        return self.pcm_frame_bytes  # no compression
