"""Text-to-speech providers."""

from collections.abc import AsyncIterator

import numpy as np

from ovi_voice_assistant.tts.tts import TTS

__all__ = ["TTS", "resample", "split_sentences"]


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 16-bit PCM audio with proper anti-aliasing."""
    import audresample

    # audresample expects float32 [channels, samples]
    signal = audio.astype(np.float32) / 32768.0
    signal = signal.reshape(1, -1)
    resampled = audresample.resample(signal, src_rate, dst_rate)
    return (resampled[0] * 32768.0).clip(-32768, 32767).astype(np.int16)


_LISTEN_TOKEN = "[LISTEN]"
_SENTENCE_ENDINGS = ".!?"


async def split_sentences(text_chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Split a stream of text tokens into sentences at .!? boundaries.

    Strips [LISTEN] control tokens from output.
    """
    buf = ""

    async for chunk in text_chunks:
        buf += chunk

        while True:
            split_idx = -1
            for i, ch in enumerate(buf):
                if (
                    ch in _SENTENCE_ENDINGS
                    and i > 10
                    and (i + 1 >= len(buf) or buf[i + 1] == " ")
                ):
                    split_idx = i + 1
                    break

            if split_idx == -1:
                break

            sentence = buf[:split_idx].strip().replace(_LISTEN_TOKEN, "").strip()
            buf = buf[split_idx:].lstrip()

            if sentence:
                yield sentence

    remainder = buf.strip().replace(_LISTEN_TOKEN, "").strip()
    if remainder:
        yield remainder
