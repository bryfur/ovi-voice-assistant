"""Tests for the Nemotron STT module."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import numpy as np
import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.stt.nemotron_stt import (
    BLANK_ID,
    DEFAULT_MODEL,
    MAX_LISTEN_S,
    MAX_SYMBOLS_PER_FRAME,
    NO_SPEECH_TIMEOUT_S,
    SILENCE_CHUNKS_THRESHOLD,
    NemotronSTT,
)


@pytest.fixture
def settings():
    return Settings(_env_file=None, devices="", openai_api_key="test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = ["<unk>", "\u2581Hello", "\u2581world", "\u2581how", "\u2581are", "\u2581you"]


def _mock_encoder(token_schedule: list[list[int]] | None = None):
    """Create a mock ONNX encoder session.

    Args:
        token_schedule: ignored here (encoder doesn't produce tokens),
            but controls how many frames per step.
    """
    session = MagicMock(spec=["run"])

    call_idx = 0

    def _run(_output_names, inputs):
        nonlocal call_idx
        call_idx += 1
        n_frames = 4  # simplified: 4 encoded frames per chunk
        return [
            np.zeros((1, 1024, n_frames), dtype=np.float32),  # encoded
            np.array([n_frames], dtype=np.int64),  # encoded_lengths
            inputs["cache_last_channel"],  # cache passthrough
            inputs["cache_last_time"],
            inputs["cache_last_channel_len"],
        ]

    session.run.side_effect = _run
    return session


def _mock_decoder(token_schedule: list[list[int]]):
    """Create a mock ONNX decoder session.

    Args:
        token_schedule: list of lists. Each inner list is the sequence of
            tokens for one encoder frame. Empty list or [BLANK_ID] → blank.
            The decoder cycles through the schedule.
    """
    session = MagicMock(spec=["run"])
    frame_idx = 0
    symbol_idx = 0

    def _run(_output_names, inputs):
        nonlocal frame_idx, symbol_idx

        schedule_idx = frame_idx % len(token_schedule) if token_schedule else 0
        frame_tokens = token_schedule[schedule_idx] if token_schedule else []

        if symbol_idx < len(frame_tokens):
            token_id = frame_tokens[symbol_idx]
            symbol_idx += 1
        else:
            token_id = BLANK_ID
            # Move to next frame
            frame_idx += 1
            symbol_idx = 0

        logits = np.zeros((1, 1, 1, BLANK_ID + 1), dtype=np.float32)
        logits[0, 0, 0, token_id] = 10.0  # high logit for target token

        return [
            logits,
            np.array([1], dtype=np.int32),  # prednet_lengths
            inputs["input_states_1"],  # states passthrough
            inputs["input_states_2"],
        ]

    session.run.side_effect = _run
    return session


def _build_stt(settings, token_schedule=None):
    """Build a NemotronSTT with mocked ONNX sessions."""
    stt = NemotronSTT(settings)
    stt._encoder = _mock_encoder()
    stt._decoder = _mock_decoder(token_schedule or [])
    stt._vocab = VOCAB
    stt._filterbank = np.random.randn(128, 257).astype(np.float32)
    stt._window = np.hanning(400).astype(np.float32)
    stt._chunk_samples = 8960
    stt._chunk_mel_frames = 56
    stt._pre_encode_cache_frames = 9
    return stt


def _pcm_chunk(n_samples: int = 8960) -> bytes:
    """Return n_samples of 16-bit silence at 16kHz."""
    return b"\x00" * (n_samples * 2)


async def _async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_model(self):
        assert DEFAULT_MODEL == "danielbodart/nemotron-speech-600m-onnx"

    def test_blank_id(self):
        assert BLANK_ID == 1024

    def test_max_symbols(self):
        assert MAX_SYMBOLS_PER_FRAME == 10

    def test_no_speech_timeout(self):
        assert NO_SPEECH_TIMEOUT_S == 5.0

    def test_max_listen(self):
        assert MAX_LISTEN_S == 60.0

    def test_silence_threshold(self):
        assert SILENCE_CHUNKS_THRESHOLD == 2


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_model_when_no_slash(self, settings):
        settings.stt_model = "base.en"

        stt = NemotronSTT(settings)

        assert stt._model_name == DEFAULT_MODEL

    def test_keeps_hf_repo(self, settings):
        settings.stt_model = "danielbodart/nemotron-speech-600m-onnx"

        stt = NemotronSTT(settings)

        assert stt._model_name == "danielbodart/nemotron-speech-600m-onnx"

    def test_raises_without_load(self, settings):
        stt = NemotronSTT(settings)

        with pytest.raises(RuntimeError, match="load"):
            stt.transcribe(b"\x00" * 100)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------


class TestPreemph:
    def test_basic_preemph(self, settings):
        stt = NemotronSTT(settings)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result, last = stt._apply_preemph(audio, prev_sample=0.0)

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0 - 0.97 * 1.0)
        assert result[2] == pytest.approx(3.0 - 0.97 * 2.0)
        assert last == pytest.approx(3.0)

    def test_preemph_with_prev_sample(self, settings):
        stt = NemotronSTT(settings)
        audio = np.array([1.0], dtype=np.float32)

        result, _ = stt._apply_preemph(audio, prev_sample=0.5)

        assert result[0] == pytest.approx(1.0 - 0.97 * 0.5)


class TestComputeMel:
    def test_output_shape(self, settings):
        stt = _build_stt(settings)
        audio = np.random.randn(9200).astype(np.float32)

        mel = stt._compute_mel(audio)

        assert mel.shape[0] == 128
        assert mel.shape[1] == 56  # (9200 - 400) // 160 + 1

    def test_empty_audio(self, settings):
        stt = _build_stt(settings)

        mel = stt._compute_mel(np.array([], dtype=np.float32))

        assert mel.shape == (128, 0)

    def test_short_audio(self, settings):
        stt = _build_stt(settings)
        audio = np.zeros(100, dtype=np.float32)

        mel = stt._compute_mel(audio)

        assert mel.shape[1] == 0  # too short for any frame


class TestDecodeTokens:
    def test_basic_decode(self, settings):
        stt = _build_stt(settings)

        result = stt._decode_tokens([1, 2])

        assert result == "Hello world"

    def test_empty_tokens(self, settings):
        stt = _build_stt(settings)

        result = stt._decode_tokens([])

        assert result == ""

    def test_out_of_range_skipped(self, settings):
        stt = _build_stt(settings)

        result = stt._decode_tokens([1, 9999, 2])

        assert result == "Hello world"


# ---------------------------------------------------------------------------
# transcribe (sync, mocked)
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_produces_text(self, settings):
        # Each encoder step produces 4 frames; decoder emits token 1 for each
        stt = _build_stt(settings, token_schedule=[[1]])
        audio = np.zeros(8960, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert "Hello" in result

    def test_short_audio_returns_empty(self, settings):
        stt = _build_stt(settings, token_schedule=[[1]])
        audio = np.zeros(100, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert result == ""

    def test_all_blanks_returns_empty(self, settings):
        stt = _build_stt(settings, token_schedule=[])
        audio = np.zeros(8960, dtype=np.int16).tobytes()

        result = stt.transcribe(audio)

        assert result == ""


# ---------------------------------------------------------------------------
# transcribe_stream (async, mocked)
# ---------------------------------------------------------------------------


class TestTranscribeStream:
    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty(self, settings):
        stt = _build_stt(settings, token_schedule=[])

        result = await stt.transcribe_stream(_async_chunks([]))

        assert result == ""

    @pytest.mark.asyncio
    async def test_speech_then_silence_returns_text(self, settings):
        # First chunk: emit tokens; next chunks: all blank → end of speech
        stt = _build_stt(settings, token_schedule=[[1]])
        # Need 1 speech chunk + SILENCE_CHUNKS_THRESHOLD silent chunks
        n_chunks = 1 + SILENCE_CHUNKS_THRESHOLD + 1
        chunks = [_pcm_chunk() for _ in range(n_chunks)]

        # After first chunk produces tokens, switch to all-blank
        call_count = 0
        orig_run = stt._decoder.run.side_effect

        def switching_decoder(_output_names, inputs):
            nonlocal call_count
            call_count += 1
            # First ~20 calls (from first encoder chunk): emit token 1
            # After that: only blanks
            if call_count <= 8:
                return orig_run(_output_names, inputs)
            logits = np.zeros((1, 1, 1, BLANK_ID + 1), dtype=np.float32)
            logits[0, 0, 0, BLANK_ID] = 10.0
            return [
                logits,
                np.array([1], dtype=np.int32),
                inputs["input_states_1"],
                inputs["input_states_2"],
            ]

        stt._decoder.run.side_effect = switching_decoder

        result = await stt.transcribe_stream(_async_chunks(chunks))

        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_no_speech_returns_empty(self, settings):
        stt = _build_stt(settings, token_schedule=[])
        chunks = [_pcm_chunk() for _ in range(3)]

        result = await stt.transcribe_stream(_async_chunks(chunks))

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_callback_does_not_error(self, settings):
        stt = _build_stt(settings, token_schedule=[[1]])
        chunks = [_pcm_chunk() for _ in range(3)]

        result = await stt.transcribe_stream(_async_chunks(chunks), on_vad_start=None)

        assert isinstance(result, str)
