"""Tests for ovi_voice_assistant.tts.piper_tts."""

from collections.abc import AsyncIterator

import pytest

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.tts.piper_tts import PiperTTS


@pytest.fixture
def settings():
    return Settings(_env_file=None, devices="", openai_api_key="test-key")


class MockPiperTTS(PiperTTS):
    """PiperTTS subclass that records synthesize calls instead of using piper."""

    def __init__(self):
        # Bypass parent __init__ which requires a Settings object.
        self._voice = None
        self._target_rate = 16000
        self.sample_rate = 16000
        self.sample_width = 2
        self.channels = 1
        self._sentences: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self._sentences.append(text)
        return b"\x00" * 100  # dummy PCM


async def _chunks(*texts: str) -> AsyncIterator[str]:
    for t in texts:
        yield t


async def _collect_stream(
    tts: MockPiperTTS, text_chunks: AsyncIterator[str]
) -> list[bytes]:
    results: list[bytes] = []
    async for audio in tts.synthesize_stream(text_chunks):
        results.append(audio)
    return results


class TestPiperTTSInit:
    def test_defaults(self, settings):
        tts = PiperTTS(settings)

        assert tts.sample_rate == 16000
        assert tts.sample_width == 2
        assert tts.channels == 1

    def test_custom_sample_rate(self, settings):
        tts = PiperTTS(settings, sample_rate=22050)

        assert tts.sample_rate == 22050


class TestSynthesizeStream:
    @pytest.mark.asyncio
    async def test_single_sentence_with_period(self):
        tts = MockPiperTTS()

        await _collect_stream(tts, _chunks("Hello world this is a test."))

        assert tts._sentences == ["Hello world this is a test."]

    @pytest.mark.asyncio
    async def test_two_sentences(self):
        tts = MockPiperTTS()

        await _collect_stream(
            tts,
            _chunks("Hello world this is. Another sentence here."),
        )

        assert len(tts._sentences) == 2
        assert tts._sentences[0] == "Hello world this is."
        assert tts._sentences[1] == "Another sentence here."

    @pytest.mark.asyncio
    async def test_short_sentence_no_split(self):
        """A period before index 10 should NOT trigger a split."""
        tts = MockPiperTTS()

        await _collect_stream(tts, _chunks("Hi. Ok."))

        assert tts._sentences == ["Hi. Ok."]

    @pytest.mark.asyncio
    async def test_exclamation_splits(self):
        tts = MockPiperTTS()

        await _collect_stream(
            tts,
            _chunks("This is exciting! And this too."),
        )

        assert len(tts._sentences) == 2
        assert tts._sentences[0] == "This is exciting!"

    @pytest.mark.asyncio
    async def test_question_mark_splits(self):
        tts = MockPiperTTS()

        await _collect_stream(
            tts,
            _chunks("Is this working? Yes it really is."),
        )

        assert len(tts._sentences) == 2
        assert tts._sentences[0] == "Is this working?"

    @pytest.mark.asyncio
    async def test_listen_token_stripped(self):
        tts = MockPiperTTS()

        await _collect_stream(
            tts,
            _chunks("Hello world this is a [LISTEN] test."),
        )

        assert all("[LISTEN]" not in s for s in tts._sentences)

    @pytest.mark.asyncio
    async def test_listen_token_stripped_from_remainder(self):
        tts = MockPiperTTS()

        await _collect_stream(tts, _chunks("short [LISTEN]"))

        assert tts._sentences == ["short"]

    @pytest.mark.asyncio
    async def test_empty_chunks_produce_no_output(self):
        tts = MockPiperTTS()

        results = await _collect_stream(tts, _chunks("", "", ""))

        assert results == []
        assert tts._sentences == []

    @pytest.mark.asyncio
    async def test_no_sentence_ending_goes_to_remainder(self):
        tts = MockPiperTTS()

        await _collect_stream(tts, _chunks("hello world no ending"))

        assert tts._sentences == ["hello world no ending"]

    @pytest.mark.asyncio
    async def test_incremental_chunks(self):
        tts = MockPiperTTS()
        parts = ["This is a ", "long sentence. ", "Here is another one."]

        await _collect_stream(tts, _chunks(*parts))

        assert len(tts._sentences) == 2
        assert tts._sentences[0] == "This is a long sentence."
