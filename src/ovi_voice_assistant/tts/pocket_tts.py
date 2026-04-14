"""Streaming text-to-speech using pocket-tts (Kyutai)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import cast

import numpy as np

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.tts import resample, split_sentences
from ovi_voice_assistant.tts.tts import TTS

logger = logging.getLogger(__name__)

# Noise gate threshold (fraction of peak). Samples below this are zeroed.
_GATE_THRESHOLD = 0.01


class PocketTTS(TTS):
    """Text-to-speech engine using Kyutai pocket-tts with true chunk streaming."""

    def __init__(self, settings: Settings, sample_rate: int = 16000) -> None:
        self._settings = settings
        self._model = None
        self._voice_state = None
        self._target_rate = sample_rate  # 0 = native model rate
        self.sample_rate: int = sample_rate
        self.sample_width: int = 2  # 16-bit
        self.channels: int = 1  # mono

    def load(self) -> None:
        from pocket_tts import TTSModel

        logger.info("Loading pocket-tts model")
        self._model = TTSModel.load_model()
        native_rate = self._model.sample_rate
        if self._target_rate == 0:
            self.sample_rate = native_rate
        logger.info(
            "pocket-tts model loaded (native %dHz, output %dHz)",
            native_rate,
            self.sample_rate,
        )

        voice = self._settings.tts_model or "alba"
        logger.info("Loading voice prompt: %s", voice)
        self._voice_state = self._model.get_state_for_audio_prompt(voice)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw 16-bit PCM at self.sample_rate."""
        if self._model is None or self._voice_state is None:
            raise RuntimeError("Call load() first")

        audio_tensor = self._model.generate_audio(self._voice_state, text)
        samples = audio_tensor.numpy()
        peak = np.abs(samples).max()
        if peak > 0:
            samples = samples * (0.9 / peak)
        samples[np.abs(samples) < _GATE_THRESHOLD] = 0.0
        audio = (samples * 32767).clip(-32768, 32767).astype(np.int16)

        native_rate = self._model.sample_rate
        if native_rate != self.sample_rate:
            audio = resample(audio, native_rate, self.sample_rate)

        return audio.tobytes()

    async def synthesize_stream(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Stream TTS: split tokens into sentences, stream each chunk."""
        loop = asyncio.get_running_loop()
        async for sentence in split_sentences(text_chunks):
            logger.debug("TTS streaming: %r", sentence[:60])
            async for pcm in self._stream_chunks(loop, sentence):
                yield pcm

    def _iter_stream(self, text: str):
        """Blocking generator: yields PCM bytes via generate_audio_stream."""
        native_rate = self._model.sample_rate
        running_peak = 0.0
        for audio_tensor in self._model.generate_audio_stream(self._voice_state, text):
            samples = audio_tensor.numpy()
            chunk_peak = np.abs(samples).max()
            if chunk_peak > running_peak:
                running_peak = chunk_peak
            if running_peak > 0:
                samples = samples * (0.9 / running_peak)
            # Noise gate: zero out samples below threshold to kill inter-sentence static
            samples[np.abs(samples) < _GATE_THRESHOLD] = 0.0
            audio = (samples * 32767).clip(-32768, 32767).astype(np.int16)
            if native_rate != self.sample_rate:
                audio = resample(audio, native_rate, self.sample_rate)
            yield audio.tobytes()

    async def _stream_chunks(
        self, loop: asyncio.AbstractEventLoop, text: str
    ) -> AsyncIterator[bytes]:
        """Bridge blocking generate_audio_stream to async yields."""
        _sentinel = object()
        it = self._iter_stream(text)
        while True:
            chunk = await loop.run_in_executor(None, next, it, _sentinel)
            if chunk is _sentinel:
                break
            yield cast(bytes, chunk)
