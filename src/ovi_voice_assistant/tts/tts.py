"""Text-to-speech base class."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator

from ovi_voice_assistant.config import Settings

logger = logging.getLogger(__name__)

_ITER_DONE = object()


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

    def synthesize_iter(self, text: str) -> Iterable[bytes]:
        """Yield PCM chunks for text as they become available.

        Default implementation falls back to ``synthesize`` and yields the
        whole buffer once. Providers that can emit sub-sentence audio (e.g.
        Kokoro's per-phoneme-batch inference) should override this to yield
        each sub-chunk the moment it's ready.
        """
        audio = self.synthesize(text)
        if audio:
            yield audio

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream TTS with pipelined synthesis and playback.

        A background producer synthesizes sentences into a small bounded
        queue while the consumer yields ready audio to the device. The
        producer pulls PCM sub-chunks from ``synthesize_iter`` one at a
        time through the executor so that providers emitting per-batch
        audio can feed the device mid-sentence.
        """
        from ovi_voice_assistant.tts import split_sentences

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None | Exception] = asyncio.Queue(maxsize=2)

        def _next_chunk(it: Iterator[bytes]) -> bytes | object:
            try:
                return next(it)
            except StopIteration:
                return _ITER_DONE

        async def producer() -> None:
            try:
                async for sentence in split_sentences(text_chunks):
                    logger.debug("TTS synthesizing: %r", sentence[:60])
                    it = iter(
                        await loop.run_in_executor(None, self.synthesize_iter, sentence)
                    )
                    while True:
                        chunk = await loop.run_in_executor(None, _next_chunk, it)
                        if chunk is _ITER_DONE:
                            break
                        if chunk:
                            await queue.put(chunk)  # type: ignore[arg-type]
            except Exception as e:
                await queue.put(e)
                return
            await queue.put(None)

        task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
