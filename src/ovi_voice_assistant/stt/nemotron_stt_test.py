"""Tests for the Nemotron STT module."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.stt.nemotron_stt import (
    BLANK_ID,
    MAX_LISTEN_S,
    MAX_SYMBOLS_PER_FRAME,
    MEL_SHIFT,
    MIN_SPEECH_S,
    N_FFT,
    N_MELS,
    NO_SPEECH_TIMEOUT_S,
    SILENCE_TIMEOUT_S,
    VAD_CHUNK_SAMPLES,
    VAD_THRESHOLD,
    WIN_LENGTH,
    NemotronSTT,
)


@pytest.fixture
def settings(tmp_path):
    with patch("ovi_voice_assistant.config.CONFIG_PATH", tmp_path / "empty.yaml"):
        return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockVadSession:
    def __init__(self, probs: list[float]):
        self._probs = iter(probs)

    def run(self, _output_names, inputs):
        prob = next(self._probs, 0.0)
        return [np.float32(prob)], inputs["h"], inputs["c"]


class MockVadModel:
    def __init__(self, probs: list[float]):
        self.session = MockVadSession(probs)


def _silent_chunk(n_samples: int = VAD_CHUNK_SAMPLES) -> bytes:
    return b"\x00" * (n_samples * 2)


async def _async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def _make_encoder_output(n_enc_frames: int = 1):
    return [
        np.zeros((1, 1024, n_enc_frames), dtype=np.float32),
        np.array([n_enc_frames], dtype=np.int64),
        np.zeros((1, 24, 70, 1024), dtype=np.float32),
        np.zeros((1, 24, 1024, 8), dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    ]


def _make_decoder_output(token_id: int = BLANK_ID):
    logits = np.full((1, 1, 1, 1025), -10.0, dtype=np.float32)
    logits[0, 0, 0, token_id] = 10.0
    return [
        logits,
        np.array([1], dtype=np.int32),
        np.zeros((2, 1, 640), dtype=np.float32),
        np.zeros((2, 1, 640), dtype=np.float32),
    ]


def _setup_mel_state(stt: NemotronSTT) -> None:
    hann = np.zeros(N_FFT, dtype=np.float64)
    wo = (N_FFT - WIN_LENGTH) // 2
    i = np.arange(WIN_LENGTH, dtype=np.float64)
    hann[wo : wo + WIN_LENGTH] = 0.5 * (
        1.0 - np.cos(2.0 * np.pi * i / (WIN_LENGTH - 1))
    )
    stt._hann = hann
    stt._filterbank_f64 = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)


def _setup_mock_sessions(stt: NemotronSTT) -> None:
    stt._encoder = MagicMock()
    stt._decoder = MagicMock()
    stt._enc_in = ["audio_signal", "length", "cache_ch", "cache_time", "cache_ch_len"]
    stt._dec_in = ["enc_out", "targets", "target_len", "s1", "s2"]
    stt._encoder.run.return_value = _make_encoder_output(1)
    stt._decoder.run.return_value = _make_decoder_output(BLANK_ID)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_vad_threshold(self):
        assert VAD_THRESHOLD == 0.4

    def test_silence_timeout(self):
        assert SILENCE_TIMEOUT_S == 0.75

    def test_min_speech(self):
        assert MIN_SPEECH_S == 0.3

    def test_max_listen(self):
        assert MAX_LISTEN_S == 60.0

    def test_no_speech_timeout(self):
        assert NO_SPEECH_TIMEOUT_S == 5.0

    def test_blank_id(self):
        assert BLANK_ID == 1024


# ---------------------------------------------------------------------------
# RNNT decode
# ---------------------------------------------------------------------------


class TestDecodeChunks:
    def test_blank_only_returns_empty(self, settings):
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        mel = np.zeros((N_MELS, MEL_SHIFT + 10), dtype=np.float32)
        tokens: list[int] = []

        stt._decode_chunks(
            mel,
            0,
            np.zeros((N_MELS, 9), dtype=np.float32),
            *stt._init_caches()[:3],
            *stt._init_dec_states()[:2],
            0,
            tokens,
        )

        assert tokens == []

    def test_emits_non_blank_tokens(self, settings):
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        stt._decoder.run.side_effect = [
            _make_decoder_output(42),
            _make_decoder_output(BLANK_ID),
        ]
        mel = np.zeros((N_MELS, MEL_SHIFT + 10), dtype=np.float32)
        tokens: list[int] = []

        stt._decode_chunks(
            mel,
            0,
            np.zeros((N_MELS, 9), dtype=np.float32),
            *stt._init_caches()[:3],
            *stt._init_dec_states()[:2],
            0,
            tokens,
        )

        assert tokens == [42]

    def test_does_not_update_states_on_blank(self, settings):
        """RNNT: prediction network must NOT advance on blank."""
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        # Decoder returns non-blank(5), then blank. States should only
        # update once (for the non-blank).
        stt._decoder.run.side_effect = [
            _make_decoder_output(5),
            _make_decoder_output(BLANK_ID),
        ]
        mel = np.zeros((N_MELS, MEL_SHIFT + 10), dtype=np.float32)
        tokens: list[int] = []

        stt._decode_chunks(
            mel,
            0,
            np.zeros((N_MELS, 9), dtype=np.float32),
            *stt._init_caches()[:3],
            *stt._init_dec_states()[:2],
            0,
            tokens,
        )

        # decoder.run called twice (non-blank + blank)
        assert stt._decoder.run.call_count == 2
        assert tokens == [5]

    def test_max_symbols_per_frame_limit(self, settings):
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        stt._decoder.run.return_value = _make_decoder_output(5)
        mel = np.zeros((N_MELS, MEL_SHIFT + 10), dtype=np.float32)
        tokens: list[int] = []

        stt._decode_chunks(
            mel,
            0,
            np.zeros((N_MELS, 9), dtype=np.float32),
            *stt._init_caches()[:3],
            *stt._init_dec_states()[:2],
            0,
            tokens,
        )

        assert len(tokens) == MAX_SYMBOLS_PER_FRAME


# ---------------------------------------------------------------------------
# transcribe (batch)
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_raises_without_load(self, settings):
        stt = NemotronSTT(settings)

        with pytest.raises(RuntimeError, match="load"):
            stt.transcribe(b"\x00" * 100)

    def test_short_audio_returns_empty(self, settings):
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        short = np.zeros(1000, dtype=np.int16).tobytes()

        assert stt.transcribe(short) == ""

    def test_full_pipeline(self, settings):
        stt = NemotronSTT(settings)
        _setup_mock_sessions(stt)
        _setup_mel_state(stt)
        stt._tokens = ["\u2581hello"]
        stt._decoder.run.side_effect = [
            _make_decoder_output(0),
            _make_decoder_output(BLANK_ID),
        ]
        audio = np.zeros(9600, dtype=np.int16).tobytes()

        assert stt.transcribe(audio) == "hello"


# ---------------------------------------------------------------------------
# transcribe_stream
# ---------------------------------------------------------------------------


def _alternating_decoder():
    call = [0]

    def side_effect(*_a, **_kw):
        call[0] += 1
        return _make_decoder_output(0 if call[0] % 2 == 1 else BLANK_ID)

    return side_effect


class TestTranscribeStream:
    @pytest.mark.asyncio
    async def test_vad_start_callback_fired(self, settings):
        probs = [0.9] + [0.0] * 200
        stt = NemotronSTT(settings)
        stt._vad_model = MockVadModel(probs)
        _setup_mock_sessions(stt)
        _setup_mel_state(stt)
        callback = AsyncMock()

        await stt.transcribe_stream(
            _async_chunks([_silent_chunk() for _ in range(5)]),
            on_vad_start=callback,
        )

        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_speech_then_silence_runs_encoder(self, settings):
        speech, silence = 20, 50
        probs = [0.9] * speech + [0.0] * silence
        stt = NemotronSTT(settings)
        stt._vad_model = MockVadModel(probs)
        _setup_mock_sessions(stt)
        _setup_mel_state(stt)
        stt._tokens = ["\u2581hello"]
        stt._decoder.run.side_effect = _alternating_decoder()
        chunks = [_silent_chunk() for _ in range(speech + silence)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.1

        loop = asyncio.get_event_loop()
        with patch.object(loop, "time", side_effect=advancing_time):
            result = await stt.transcribe_stream(_async_chunks(chunks))

        assert "hello" in result
        assert stt._encoder.run.called

    @pytest.mark.asyncio
    async def test_no_speech_timeout(self, settings):
        stt = NemotronSTT(settings)
        stt._vad_model = MockVadModel([0.0] * 500)
        _setup_mock_sessions(stt)
        chunks = [_silent_chunk() for _ in range(200)]
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.5

        loop = asyncio.get_event_loop()
        with patch.object(loop, "time", side_effect=advancing_time):
            result = await stt.transcribe_stream(_async_chunks(chunks))

        assert result == ""
        stt._encoder.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_stream(self, settings):
        stt = NemotronSTT(settings)
        stt._vad_model = MockVadModel([])
        _setup_mock_sessions(stt)

        assert await stt.transcribe_stream(_async_chunks([])) == ""

    @pytest.mark.asyncio
    async def test_no_callback_ok(self, settings):
        probs = [0.9] + [0.0] * 50
        stt = NemotronSTT(settings)
        stt._vad_model = MockVadModel(probs)
        _setup_mock_sessions(stt)
        _setup_mel_state(stt)
        call_count = 0
        base_time = 1000.0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            return base_time + call_count * 0.5

        loop = asyncio.get_event_loop()
        with patch.object(loop, "time", side_effect=advancing_time):
            result = await stt.transcribe_stream(
                _async_chunks([_silent_chunk() for _ in range(5)]),
                on_vad_start=None,
            )

        assert isinstance(result, str)
