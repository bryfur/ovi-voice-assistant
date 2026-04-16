"""Single-worker queue that serializes TTS synthesis and device playback.

Multiple callers (say tool, response TTS, announcements) submit text and
return immediately. One worker processes submissions in FIFO order,
streaming audio to the output so utterances never overlap on the wire.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.tts import TTS

logger = logging.getLogger(__name__)


class SpeechQueue:
    def __init__(self, tts: TTS, output: PipelineOutput) -> None:
        self._tts = tts
        self._output = output
        self._queue: asyncio.Queue[
            tuple[AsyncIterator[str], asyncio.Future[None]] | None
        ] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    def submit(self, text: str) -> asyncio.Future[None]:
        """Enqueue a fixed utterance. Returns a future that resolves once played."""

        async def tokens() -> AsyncIterator[str]:
            yield text

        return self.submit_stream(tokens())

    def submit_stream(self, tokens: AsyncIterator[str]) -> asyncio.Future[None]:
        self.start()
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((tokens, done))
        return done

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            tokens, done = item
            try:
                async for pcm in self._tts.synthesize_stream(tokens):
                    await self._output.send_audio(pcm)
                if not done.done():
                    done.set_result(None)
            except asyncio.CancelledError:
                if not done.done():
                    done.cancel()
                raise
            except Exception as e:
                logger.exception("SpeechQueue worker error")
                if not done.done():
                    done.set_exception(e)
