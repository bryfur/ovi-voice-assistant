"""Audio codec abstraction for Ovi -- PCM, LC3, Opus."""

import logging

from ovi_voice_assistant.codec.audio_codec import AudioCodec, CodecType

logger = logging.getLogger(__name__)

__all__ = ["AudioCodec", "CodecType", "create_codec"]

LC3_VALID_RATES = [8000, 16000, 24000, 32000, 48000]
OPUS_VALID_RATES = [8000, 12000, 16000, 24000, 48000]


def _nearest_valid_rate(rate: int, valid_rates: list[int]) -> int:
    """Return the closest supported sample rate."""
    return min(valid_rates, key=lambda r: abs(r - rate))


def create_codec(
    codec_type: CodecType | str,
    sample_rate: int,
    channels: int = 1,
    nbyte: int | None = None,
) -> AudioCodec:
    """Factory for codec instances."""
    if isinstance(codec_type, str):
        codec_type = CodecType(codec_type)
    if codec_type == CodecType.PCM:
        from ovi_voice_assistant.codec.pcm import PcmCodec

        return PcmCodec(sample_rate, channels)
    elif codec_type == CodecType.LC3:
        valid_rate = _nearest_valid_rate(sample_rate, LC3_VALID_RATES)
        if valid_rate != sample_rate:
            logger.warning(
                "LC3 does not support %dHz, using nearest valid rate %dHz",
                sample_rate,
                valid_rate,
            )
        from ovi_voice_assistant.codec.lc3 import Lc3Codec

        if nbyte is not None:
            return Lc3Codec(valid_rate, channels, nbyte=nbyte)
        return Lc3Codec(valid_rate, channels)
    elif codec_type == CodecType.OPUS:
        valid_rate = _nearest_valid_rate(sample_rate, OPUS_VALID_RATES)
        if valid_rate != sample_rate:
            logger.warning(
                "Opus does not support %dHz, using nearest valid rate %dHz",
                sample_rate,
                valid_rate,
            )
        from ovi_voice_assistant.codec.opus import OpusCodec

        return OpusCodec(valid_rate, channels)
    raise ValueError(f"Unknown codec: {codec_type}")
