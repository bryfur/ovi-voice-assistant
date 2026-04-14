"""Opus codec wrapper using opuslib."""

from ovi_voice_assistant.codec.audio_codec import AudioCodec, CodecType

OPUS_FRAME_DURATION_MS = 20
OPUS_MAX_ENCODED_BYTES = 80  # reasonable max at ~32kbps for 20ms VBR


class OpusCodec(AudioCodec):
    """Opus audio codec (20ms frames, VBR)."""

    codec_type = CodecType.OPUS
    codec_id = 2

    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        import opuslib

        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = sample_rate * OPUS_FRAME_DURATION_MS // 1000

        self._encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_VOIP)
        self._decoder = opuslib.Decoder(sample_rate, channels)

    def encode(self, pcm: bytes) -> bytes:
        return self._encoder.encode(pcm, self._frame_samples)

    def decode(self, data: bytes) -> bytes:
        return self._decoder.decode(data, self._frame_samples)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        return OPUS_FRAME_DURATION_MS

    @property
    def pcm_frame_bytes(self) -> int:
        return self._frame_samples * self._channels * 2

    @property
    def encoded_frame_bytes(self) -> int:
        return OPUS_MAX_ENCODED_BYTES  # VBR; this is a reasonable upper bound
