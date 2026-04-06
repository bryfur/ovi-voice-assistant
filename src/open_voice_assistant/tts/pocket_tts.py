"""Streaming text-to-speech using pocket-tts (Kyutai)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import cast

import numpy as np

from open_voice_assistant.config import Settings
from open_voice_assistant.tts import TTS, resample

logger = logging.getLogger(__name__)


class PocketTTS(TTS):
    """Text-to-speech engine using Kyutai pocket-tts with true chunk streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._voice_state = None
        self.sample_rate: int = 16000  # output rate (resampled if needed)
        self.sample_width: int = 2  # 16-bit
        self.channels: int = 1  # mono

    def load(self) -> None:
        from pocket_tts import TTSModel

        logger.info("Loading pocket-tts model")
        self._model = TTSModel.load_model()
        native_rate = self._model.sample_rate
        logger.info("pocket-tts model loaded (native %dHz, output %dHz)",
                     native_rate, self.sample_rate)

        voice = self._settings.tts_model or "alba"
        logger.info("Loading voice prompt: %s", voice)
        self._voice_state = self._model.get_state_for_audio_prompt(voice)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate."""
        if self._model is None or self._voice_state is None:
            raise RuntimeError("Call load() first")

        audio_tensor = self._model.generate_audio(self._voice_state, text)
        audio = (audio_tensor.numpy() * 32767).clip(-32768, 32767).astype(np.int16)

        native_rate = self._model.sample_rate
        if native_rate != self.sample_rate:
            audio = resample(audio, native_rate, self.sample_rate)

        return audio.tobytes()

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Stream TTS: accumulate text into sentences, synthesize each eagerly."""
        sentence_buf = ""
        full_response = ""
        sentence_endings = ".!?"
        loop = asyncio.get_running_loop()

        async for chunk in text_chunks:
            sentence_buf += chunk
            if len(full_response) < 200:
                full_response += chunk

            while True:
                split_idx = -1
                for i, ch in enumerate(sentence_buf):
                    if ch in sentence_endings and i > 10:
                        if i + 1 >= len(sentence_buf) or sentence_buf[i + 1] == " ":
                            split_idx = i + 1
                            break

                if split_idx == -1:
                    break

                sentence = sentence_buf[:split_idx].strip()
                sentence_buf = sentence_buf[split_idx:].lstrip()

                sentence = sentence.replace("[LISTEN]", "").strip()
                if sentence:
                    logger.debug("TTS streaming: %r", sentence[:60])
                    async for pcm in self._stream_chunks(loop, sentence):
                        yield pcm

        remainder = sentence_buf.strip()
        remainder = remainder.replace("[LISTEN]", "").strip()
        if remainder:
            logger.debug("TTS streaming remainder: %r", remainder[:60])
            async for pcm in self._stream_chunks(loop, remainder):
                yield pcm

        logger.debug("Agent response: %r", full_response[:200])

    def _iter_stream(self, text: str):
        """Blocking generator: yields PCM bytes via generate_audio_stream."""
        native_rate = self._model.sample_rate
        for audio_tensor in self._model.generate_audio_stream(self._voice_state, text):
            audio = (audio_tensor.numpy() * 32767).clip(-32768, 32767).astype(np.int16)
            if native_rate != self.sample_rate:
                audio = resample(audio, native_rate, self.sample_rate)
            yield audio.tobytes()

    async def _stream_chunks(self, loop: asyncio.AbstractEventLoop, text: str) -> AsyncIterator[bytes]:
        """Bridge blocking generate_audio_stream to async yields."""
        _sentinel = object()
        it = self._iter_stream(text)
        while True:
            chunk = await loop.run_in_executor(None, next, it, _sentinel)
            if chunk is _sentinel:
                break
            yield cast(bytes, chunk)

