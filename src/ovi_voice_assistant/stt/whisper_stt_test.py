"""Tests for the STT / Whisper module."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.stt.whisper_stt import (
    MAX_LISTEN_S,
    MIN_SPEECH_S,
    NO_SPEECH_TIMEOUT_S,
    SILENCE_TIMEOUT_S,
    VAD_CHUNK_SAMPLES,
    VAD_THRESHOLD,
    WhisperSTT,
)


@pytest.fixture
def settings():
    return Settings(_env_file=None, devices="", openai_api_key="test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockVadSession:
    """Yields pre-programmed speech probabilities one per VAD chunk."""

    def __init__(self, probs: list[float]):
        self._probs = iter(probs)

    def run(self, _output_names, inputs):
        prob = next(self._probs, 0.0)
        # output[0] must be convertible via float(); use a 0-d array.
        return [np.float32(prob)], inputs["h"], inputs["c"]


class MockVadModel:
    def __init__(self, probs: list[float]):
        self.session = MockVadSession(probs)


def _make_segment(text: str):
    """Return a minimal object that quacks like a faster-whisper Segment."""
    return SimpleNamespace(text=text)


def _make_info(language: str = "en", language_probability: float = 0.99):
    return SimpleNamespace(language=language, language_probability=language_probability)


def _silent_chunk(n_samples: int = VAD_CHUNK_SAMPLES) -> bytes:
    """Return n_samples of 16-bit silence (zeros)."""
    return b"\x00" * (n_samples * 2)


async def _async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Yield pre-built byte chunks as an async iterator."""
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_vad_threshold(self):
        assert VAD_THRESHOLD == 0.4

    def test_silence_timeout(self):
        assert SILENCE_TIMEOUT_S == 1.0

    def test_min_speech(self):
        assert MIN_SPEECH_S == 0.3

    def test_max_listen(self):
        assert MAX_LISTEN_S == 60.0

    def test_no_speech_timeout(self):
        assert NO_SPEECH_TIMEOUT_S == 5.0

    def test_vad_chunk_samples(self):
        assert VAD_CHUNK_SAMPLES == 512


# ---------------------------------------------------------------------------
# Audio normalisation
# ---------------------------------------------------------------------------


class TestAudioNormalization:
    def test_int16_to_float32_range(self):
        """int16 values divided by 32768.0 should produce floats in [-1, 1)."""
        raw = np.array([0, 32767, -32768, 1000, -1000], dtype=np.int16)

        audio = raw.astype(np.float32) / 32768.0

        assert audio.dtype == np.float32
        assert audio[0] == pytest.approx(0.0)
        assert audio[1] == pytest.approx(32767 / 32768.0)
        assert audio[2] == pytest.approx(-1.0)
        assert float(audio.min()) >= -1.0 and float(audio.max()) < 1.0

    def test_silence_is_zero(self):
        raw = np.zeros(100, dtype=np.int16)

        audio = raw.astype(np.float32) / 32768.0

        np.testing.assert_array_equal(audio, np.zeros(100, dtype=np.float32))


# ---------------------------------------------------------------------------
# transcribe (sync, mocked model)
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_raises_without_load(self, settings):
        stt = WhisperSTT(settings)
        with pytest.raises(RuntimeError, match="load"):
            stt.transcribe(b"\x00" * 100)

    def test_short_audio_returns_empty(self, settings):
        """Audio shorter than 0.1 s should return '' immediately."""
        stt = WhisperSTT(settings)
        stt._model = MagicMock()  # pretend loaded
        # 0.1 s at 16 kHz = 1600 samples, need fewer than that
        short_audio = np.zeros(1000, dtype=np.int16).tobytes()

        result = stt.transcribe(short_audio)

        assert result == ""
        # Model.transcribe should NOT have been called
        stt._model.transcribe.assert_not_called()

    def test_segments_joined(self, settings):
        stt = WhisperSTT(settings)
        mock_model = MagicMock()
        segments = [_make_segment(" hello "), _make_segment(" world ")]
        mock_model.transcribe.return_value = (iter(segments), _make_info())
        stt._model = mock_model
        # 0.5 s of silence (enough samples to pass the length gate)
        audio = np.zeros(8000, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert result == "hello world"

    def test_single_segment(self, settings):
        stt = WhisperSTT(settings)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([_make_segment("  testing  ")]),
            _make_info(),
        )
        stt._model = mock_model
        audio = np.zeros(8000, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert result == "testing"

    def test_no_segments_returns_empty(self, settings):
        stt = WhisperSTT(settings)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), _make_info())
        stt._model = mock_model
        audio = np.zeros(8000, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert result == ""

    def test_transcribe_passes_correct_params(self, settings):
        stt = WhisperSTT(settings)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), _make_info())
        stt._model = mock_model
        audio = np.zeros(8000, dtype=np.int16).tobytes()

        stt.transcribe(audio)

        _args, kwargs = mock_model.transcribe.call_args
        assert kwargs["language"] == "en"
        assert kwargs["beam_size"] == 1
        assert kwargs["vad_filter"] is True
        assert kwargs["condition_on_previous_text"] is False


# ---------------------------------------------------------------------------
# transcribe_stream (async, mocked VAD + transcribe)
# ---------------------------------------------------------------------------


class TestTranscribeStream:
    @pytest.mark.asyncio
    async def test_vad_start_callback_fired(self, settings):
        """on_vad_start should be awaited when speech probability exceeds threshold."""
        probs = [0.9] + [0.0] * 200
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel(probs)
        stt._model = MagicMock()
        callback = AsyncMock()
        chunks = [_silent_chunk() for _ in range(5)]

        with patch.object(stt, "transcribe", return_value="hi"):
            await stt.transcribe_stream(_async_chunks(chunks), on_vad_start=callback)

        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_speech_then_silence_triggers_transcription(self, settings):
        """After speech + enough silence, transcribe should be called."""
        speech_chunks = 15
        silence_chunks = 50
        probs = [0.9] * speech_chunks + [0.0] * silence_chunks
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel(probs)
        stt._model = MagicMock()
        total_chunks = speech_chunks + silence_chunks
        chunks = [_silent_chunk() for _ in range(total_chunks)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.1

        loop = asyncio.get_event_loop()
        with (
            patch.object(loop, "time", side_effect=advancing_time),
            patch.object(stt, "transcribe", return_value="hello there") as mock_tx,
        ):
            result = await stt.transcribe_stream(_async_chunks(chunks))

        assert result == "hello there"
        mock_tx.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_speech_timeout_returns_empty(self, settings):
        """If no speech is detected within NO_SPEECH_TIMEOUT_S, return ''."""
        probs = [0.0] * 500
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel(probs)
        stt._model = MagicMock()
        chunks = [_silent_chunk() for _ in range(200)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.5

        loop = asyncio.get_event_loop()
        with (
            patch.object(loop, "time", side_effect=advancing_time),
            patch.object(stt, "transcribe", return_value=""),
        ):
            result = await stt.transcribe_stream(_async_chunks(chunks))

        assert result == ""

    @pytest.mark.asyncio
    async def test_short_speech_resets_state(self, settings):
        """Speech shorter than MIN_SPEECH_S followed by silence should reset."""
        short_speech = 1
        silence_after_short = 30
        real_speech = 15
        silence_after_real = 30
        probs = (
            [0.9] * short_speech
            + [0.0] * silence_after_short
            + [0.9] * real_speech
            + [0.0] * silence_after_real
        )
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel(probs)
        stt._model = MagicMock()
        total = short_speech + silence_after_short + real_speech + silence_after_real
        chunks = [_silent_chunk() for _ in range(total)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.1

        loop = asyncio.get_event_loop()
        with (
            patch.object(loop, "time", side_effect=advancing_time),
            patch.object(stt, "transcribe", return_value="real speech") as mock_tx,
        ):
            result = await stt.transcribe_stream(_async_chunks(chunks))

        assert result == "real speech"
        mock_tx.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty(self, settings):
        """An empty async iterator should return ''."""
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel([])
        stt._model = MagicMock()

        result = await stt.transcribe_stream(_async_chunks([]))

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_callback_does_not_error(self, settings):
        """When on_vad_start is None and speech is detected, it should not raise."""
        probs = [0.9] + [0.0] * 50
        stt = WhisperSTT(settings)
        stt._vad_model = MockVadModel(probs)
        stt._model = MagicMock()
        chunks = [_silent_chunk() for _ in range(5)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.5

        loop = asyncio.get_event_loop()
        with (
            patch.object(loop, "time", side_effect=advancing_time),
            patch.object(stt, "transcribe", return_value="ok"),
        ):
            result = await stt.transcribe_stream(
                _async_chunks(chunks), on_vad_start=None
            )

        assert isinstance(result, str)
