"""Tests for VoiceAssistant class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.transport import EventType
from ovi_voice_assistant.voice_assistant import (
    LISTEN_TOKEN,
    VoiceAssistant,
    _create_stt,
    _create_tts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_va():
    """Create a VoiceAssistant with fully mocked internals."""
    settings = MagicMock()
    settings.stt.provider = "whisper"
    settings.tts.provider = "piper"

    with (
        patch("ovi_voice_assistant.voice_assistant._create_stt") as mock_stt_factory,
        patch("ovi_voice_assistant.voice_assistant._create_tts") as mock_tts_factory,
        patch("ovi_voice_assistant.voice_assistant.Assistant") as mock_assistant_cls,
    ):
        mock_stt = MagicMock()
        mock_tts = MagicMock()
        mock_agent = MagicMock()

        mock_stt_factory.return_value = mock_stt
        mock_tts_factory.return_value = mock_tts
        mock_assistant_cls.return_value = mock_agent

        va = VoiceAssistant(settings)

    return va


def _make_output():
    """Create a mock PipelineOutput."""
    return AsyncMock(spec=PipelineOutput)


async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# _create_stt / _create_tts — unknown provider
# ---------------------------------------------------------------------------


class TestCreateProviders:
    def test_create_stt_unknown_provider(self):
        settings = MagicMock()
        settings.stt.provider = "nonexistent"

        with pytest.raises(ValueError, match="Unknown STT provider"):
            _create_stt(settings)

    def test_create_tts_unknown_provider(self):
        settings = MagicMock()
        settings.tts.provider = "nonexistent"

        with pytest.raises(ValueError, match="Unknown TTS provider"):
            _create_tts(settings)


# ---------------------------------------------------------------------------
# VoiceAssistant.run — full pipeline success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_pipeline_success():
    va = _make_va()
    output = _make_output()
    captured_vad_cb = None

    async def fake_transcribe(audio_iter, on_vad_start=None):
        nonlocal captured_vad_cb
        captured_vad_cb = on_vad_start
        if on_vad_start:
            await on_vad_start()
        return "hello"

    va.stt.transcribe_stream = AsyncMock(side_effect=fake_transcribe)
    va.agent.run_streamed = MagicMock(return_value=_async_iter(["Hi ", "there"]))

    async def fake_synthesize(token_iter):
        async for _ in token_iter:
            pass
        yield b"audio"

    va.tts.synthesize_stream = MagicMock(side_effect=fake_synthesize)
    audio_in = _async_iter([b"mic_data"])

    result = await va.run(output, audio_in)

    assert result is False  # no LISTEN_TOKEN so no follow-up
    calls = output.send_event.call_args_list
    event_types = [c.args[0] for c in calls]
    assert EventType.VAD_START in event_types
    assert EventType.MIC_STOP in event_types
    assert EventType.TTS_START in event_types
    assert EventType.TTS_END in event_types
    output.send_audio.assert_called_with(b"audio")


# ---------------------------------------------------------------------------
# VoiceAssistant.run — no speech detected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_no_speech():
    va = _make_va()
    output = _make_output()
    va.stt.transcribe_stream = AsyncMock(return_value="")
    audio_in = _async_iter([b"silence"])

    result = await va.run(output, audio_in)

    assert result is False
    error_calls = [
        c for c in output.send_event.call_args_list if c.args[0] == EventType.ERROR
    ]
    assert len(error_calls) == 1
    assert b"stt-no-text" in error_calls[0].args[1]


# ---------------------------------------------------------------------------
# VoiceAssistant.run — follow-up with [LISTEN]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_follow_up_listen():
    va = _make_va()
    output = _make_output()
    va.stt.transcribe_stream = AsyncMock(return_value="hello")
    va.agent.run_streamed = MagicMock(
        return_value=_async_iter(["Sure thing. ", LISTEN_TOKEN])
    )

    async def fake_synthesize(token_iter):
        async for _ in token_iter:
            pass
        yield b"audio"

    va.tts.synthesize_stream = MagicMock(side_effect=fake_synthesize)
    audio_in = _async_iter([b"mic_data"])

    result = await va.run(output, audio_in)

    assert result is True
    event_types = [c.args[0] for c in output.send_event.call_args_list]
    assert EventType.CONTINUE in event_types


# ---------------------------------------------------------------------------
# VoiceAssistant.run — exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_exception_sends_error():
    va = _make_va()
    output = _make_output()
    va.stt.transcribe_stream = AsyncMock(side_effect=RuntimeError("boom"))
    audio_in = _async_iter([b"mic_data"])

    result = await va.run(output, audio_in)

    assert result is False
    error_calls = [
        c for c in output.send_event.call_args_list if c.args[0] == EventType.ERROR
    ]
    assert len(error_calls) == 1
    assert b"pipeline_error" in error_calls[0].args[1]


# ---------------------------------------------------------------------------
# VoiceAssistant.run — cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cancelled_no_error():
    va = _make_va()
    output = _make_output()
    va.stt.transcribe_stream = AsyncMock(side_effect=asyncio.CancelledError())
    audio_in = _async_iter([b"mic_data"])

    result = await va.run(output, audio_in)

    assert result is False
    error_calls = [
        c for c in output.send_event.call_args_list if c.args[0] == EventType.ERROR
    ]
    assert len(error_calls) == 0


# ---------------------------------------------------------------------------
# VoiceAssistant.announce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_announce():
    va = _make_va()
    output = _make_output()

    async def fake_synthesize(token_iter):
        async for _ in token_iter:
            pass
        yield b"tts_audio"

    va.tts.synthesize_stream = MagicMock(side_effect=fake_synthesize)

    await va.announce(output, "Hello world")

    calls = output.send_event.call_args_list
    event_types = [c.args[0] for c in calls]
    assert event_types[0] == EventType.TTS_START
    assert event_types[-1] == EventType.TTS_END
    output.send_audio.assert_called_with(b"tts_audio")
