"""Tests for SpeechQueue."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ovi_voice_assistant.speech_queue import SpeechQueue


async def _collect_stream(text: str):
    yield text


def _make_tts(audio: bytes = b"audio"):
    tts = MagicMock()

    async def synthesize_stream(tokens):
        async for _ in tokens:
            yield audio

    tts.synthesize_stream = synthesize_stream
    return tts


def _make_output():
    output = MagicMock()
    output.send_audio = AsyncMock()
    output.send_event = AsyncMock()
    return output


@pytest.mark.asyncio
async def test_submit_plays_text() -> None:
    tts = _make_tts(b"pcm1")
    output = _make_output()
    queue = SpeechQueue(tts, output)

    done = queue.submit("hello")
    await done

    output.send_audio.assert_awaited_once_with(b"pcm1")
    await queue.stop()


@pytest.mark.asyncio
async def test_submit_returns_before_playback() -> None:
    tts = _make_tts(b"pcm")
    output = _make_output()
    queue = SpeechQueue(tts, output)

    done = queue.submit("hi")

    assert not done.done()
    await done
    assert done.done()
    await queue.stop()


@pytest.mark.asyncio
async def test_submissions_play_in_order() -> None:
    tts = _make_tts()
    output = _make_output()
    order: list[str] = []

    async def synthesize_stream(tokens):
        async for t in tokens:
            order.append(t)
            yield t.encode()

    tts.synthesize_stream = synthesize_stream
    queue = SpeechQueue(tts, output)

    d1 = queue.submit("first")
    d2 = queue.submit("second")
    d3 = queue.submit("third")
    await d1
    await d2
    await d3

    assert order == ["first", "second", "third"]
    await queue.stop()


@pytest.mark.asyncio
async def test_stop_drains_worker() -> None:
    tts = _make_tts()
    output = _make_output()
    queue = SpeechQueue(tts, output)

    queue.submit("bye")
    await queue.stop()

    assert queue._worker is None
