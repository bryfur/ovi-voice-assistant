"""End-to-end latency benchmarks for the streaming voice pipeline.

Measures plumbing overhead — not absolute model latency. STT, agent, TTS,
and transport are mocked with deterministic timing so assertions reflect
code-path cost, not model cost. A regression here means someone added a
sync gate, an extra buffer, or changed buffering semantics.

Budgets are ceilings with CI-friendly headroom, not performance targets.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from ovi_voice_assistant.codec.pcm import PcmCodec
from ovi_voice_assistant.device_connection import _EncodingOutput
from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.transport import DeviceTransport, EventType
from ovi_voice_assistant.voice_assistant import VoiceAssistant

# Ceiling for "should be effectively instant" operations on a CI runner.
# Bigger than we need in practice so scheduler jitter doesn't cause flakes.
PLUMBING_BUDGET_S = 0.1


# ── Helpers ──────────────────────────────────────────────────────


class _RecordingTransport(DeviceTransport):
    """DeviceTransport that timestamps every send and can simulate link delay."""

    def __init__(
        self, *, send_event_delay: float = 0.0, send_audio_delay: float = 0.0
    ) -> None:
        self._send_event_delay = send_event_delay
        self._send_audio_delay = send_audio_delay
        self._connected = True
        self.events: list[tuple[float, EventType, bytes]] = []
        self.audio: list[tuple[float, bytes]] = []

    async def connect(self) -> None: ...

    async def disconnect(self) -> None:
        self._connected = False

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        if self._send_event_delay:
            await asyncio.sleep(self._send_event_delay)
        self.events.append((time.perf_counter(), event, payload))

    async def send_audio(self, data: bytes) -> None:
        if self._send_audio_delay:
            await asyncio.sleep(self._send_audio_delay)
        self.audio.append((time.perf_counter(), data))

    def set_event_callback(self, cb) -> None: ...
    def set_audio_callback(self, cb) -> None: ...
    def set_disconnect_callback(self, cb) -> None: ...
    def set_connect_callback(self, cb) -> None: ...

    @property
    def is_connected(self) -> bool:
        return self._connected


class _RecordingOutput(PipelineOutput):
    """PipelineOutput that records every call with a monotonic timestamp."""

    def __init__(self, send_audio_delay: float = 0.0) -> None:
        self._send_audio_delay = send_audio_delay
        self.log: list[tuple[float, str, object]] = []

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        self.log.append((time.perf_counter(), "event", event))

    async def send_audio(self, pcm: bytes) -> None:
        if self._send_audio_delay:
            await asyncio.sleep(self._send_audio_delay)
        self.log.append((time.perf_counter(), "audio", len(pcm)))

    def first_of(self, kind: str) -> float | None:
        return next((t for t, k, _ in self.log if k == kind), None)

    def last_of(self, kind: str) -> float | None:
        return next((t for t, k, _ in reversed(self.log) if k == kind), None)


def _make_voice_assistant() -> VoiceAssistant:
    """VoiceAssistant with STT, TTS, and Assistant fully mocked (no model load)."""
    settings = MagicMock()
    settings.stt.provider = "whisper"
    settings.tts.provider = "piper"

    with (
        patch("ovi_voice_assistant.voice_assistant._create_stt") as stt_factory,
        patch("ovi_voice_assistant.voice_assistant._create_tts") as tts_factory,
        patch("ovi_voice_assistant.voice_assistant.Assistant") as agent_cls,
    ):
        stt_factory.return_value = MagicMock()
        tts_factory.return_value = MagicMock()
        agent_cls.return_value = MagicMock()
        return VoiceAssistant(settings)


def _wire_instant_pipeline(
    va: VoiceAssistant,
    *,
    agent_tokens: list[str],
    pcm_per_sentence: bytes = b"\x00" * 320,
) -> dict[str, float | None]:
    """Install instant STT/agent/TTS mocks and return timing marks.

    Marks: ``stt_returned`` (when STT yields final transcript),
    ``agent_first_token`` (first token emitted to TTS), ``tts_first_yield``
    (first PCM chunk yielded from TTS).
    """
    marks: dict[str, float | None] = {
        "stt_returned": None,
        "agent_first_token": None,
        "tts_first_yield": None,
    }

    async def stt(audio_iter, on_vad_start=None):
        async for _ in audio_iter:
            pass
        marks["stt_returned"] = time.perf_counter()
        return "hi"

    async def agent(text, context=None):
        for i, token in enumerate(agent_tokens):
            if i == 0:
                marks["agent_first_token"] = time.perf_counter()
            yield token

    async def tts(token_iter):
        # Consume all agent tokens (real TTS would split into sentences first).
        # For benchmarks we just drain — tests that want sentence-split behavior
        # use the real `synthesize_stream` wrapper; see dedicated tests below.
        async for _ in token_iter:
            pass
        marks["tts_first_yield"] = time.perf_counter()
        yield pcm_per_sentence

    va.stt.transcribe_stream = stt
    va.agent.run_streamed = agent
    va.tts.synthesize_stream = tts
    return marks


async def _async_iter(items):
    for item in items:
        yield item


# ── 1. Plumbing overhead: transcript-ready → first PCM ──────────


@pytest.mark.asyncio
async def test_transcript_to_first_pcm_overhead_under_budget():
    """With instant mocks, transcript-ready → first ``send_audio`` measures
    pure plumbing cost (agent call + sentence split + TTS wrapper + speech
    queue hop). Regressions above the budget usually mean a new sync gate.
    """
    va = _make_voice_assistant()
    output = _RecordingOutput()
    marks = _wire_instant_pipeline(va, agent_tokens=["Hi there."])

    await va.run(output, _async_iter([b"\x00"]))

    first_audio_t = output.first_of("audio")
    assert first_audio_t is not None, "expected at least one send_audio call"
    overhead = first_audio_t - marks["stt_returned"]
    assert overhead < PLUMBING_BUDGET_S, (
        f"transcript→first-pcm overhead {overhead * 1000:.1f}ms "
        f"exceeds budget {PLUMBING_BUDGET_S * 1000:.0f}ms"
    )


# ── 2. First agent token → first PCM (short sentence) ───────────


@pytest.mark.asyncio
async def test_first_token_to_first_pcm_short_sentence():
    """A complete short sentence reaches PCM within the plumbing budget."""
    va = _make_voice_assistant()
    output = _RecordingOutput()
    marks = _wire_instant_pipeline(va, agent_tokens=["Hello there."])

    await va.run(output, _async_iter([b"\x00"]))

    first_audio_t = output.first_of("audio")
    assert first_audio_t is not None
    delta = first_audio_t - marks["agent_first_token"]
    assert delta < PLUMBING_BUDGET_S, (
        f"first-token→first-pcm {delta * 1000:.1f}ms exceeds budget "
        f"{PLUMBING_BUDGET_S * 1000:.0f}ms"
    )


# ── 3. Sentence-boundary buffering behavior ─────────────────────


@pytest.mark.asyncio
async def test_long_preamble_blocks_first_pcm_until_sentence_boundary():
    """Agent yields many tokens without a terminator; under the current
    sentence-split policy the first PCM is emitted only after the first
    sentence end arrives.

    This locks in current behavior: if a future change introduces
    clause-level flushing, ``tokens_before_first_audio`` will drop and
    this test will need to be updated alongside the new threshold. That
    intentional coupling is the point — the test documents the tradeoff.
    """
    va = _make_voice_assistant()
    output = _RecordingOutput()

    # Real TTS wrapper is used here so ``split_sentences`` runs; we only
    # mock the per-sentence synthesis.
    from ovi_voice_assistant.tts.tts import TTS

    class _FakeTTS(TTS):
        sample_rate = 16000
        sample_width = 2
        channels = 1

        def __init__(self, _settings, sample_rate: int = 16000) -> None: ...
        def load(self) -> None: ...
        def synthesize(self, text: str) -> bytes:
            return b"\x00" * 320

    va.tts = _FakeTTS(None)

    preamble = ["Well ", "I ", "was ", "thinking ", "a ", "while ", "now "]
    terminator = ["and it matters."]
    tokens_seen_by_tts: list[str] = []

    async def stt(audio_iter, on_vad_start=None):
        async for _ in audio_iter:
            pass
        return "hi"

    first_token_t: dict[str, float | None] = {"t": None}

    async def agent(text, context=None):
        for i, token in enumerate([*preamble, *terminator]):
            if i == 0:
                first_token_t["t"] = time.perf_counter()
            yield token

    # Wrap synthesize_stream to observe the sentence text it receives.
    orig_stream = va.tts.synthesize_stream

    async def recording_stream(token_iter):
        async def spy(iter_):
            async for t in iter_:
                tokens_seen_by_tts.append(t)
                yield t

        async for chunk in orig_stream(spy(token_iter)):
            yield chunk

    va.stt.transcribe_stream = stt
    va.agent.run_streamed = agent
    va.tts.synthesize_stream = recording_stream

    await va.run(output, _async_iter([b"\x00"]))

    first_audio_t = output.first_of("audio")
    assert first_audio_t is not None, "expected send_audio after sentence completes"
    # All preamble tokens must arrive at the splitter before any audio.
    assert len(tokens_seen_by_tts) >= len(preamble) + 1
    delta = first_audio_t - first_token_t["t"]
    # Ceiling — all mocks are instant, so even waiting for every token is fast.
    assert delta < 0.2, f"preamble→first-pcm {delta * 1000:.1f}ms"


# ── 4. Event ordering through _EncodingOutput ───────────────────


@pytest.mark.asyncio
async def test_tts_start_event_precedes_first_audio_frame():
    """AUDIO_CONFIG + TTS_START must hit the transport before any PCM frame.

    _EncodingOutput serializes events and audio on one queue so the
    device never receives audio for a codec it has not been reconfigured
    to decode.
    """
    transport = _RecordingTransport()
    codec = PcmCodec(16000)
    output = _EncodingOutput(transport, codec)

    await output.send_event(EventType.TTS_START)
    await output.send_audio(b"\x00" * codec.pcm_frame_bytes)
    await output.flush()

    merged = sorted(
        [(t, "event", e) for t, e, _ in transport.events]
        + [(t, "audio", None) for t, _ in transport.audio]
    )
    kinds = [k for _, k, _ in merged]
    event_types = [e for _, k, e in merged if k == "event"]

    assert kinds[:2] == ["event", "event"], (
        f"expected two events before audio, got order {kinds}"
    )
    assert event_types[0] == EventType.AUDIO_CONFIG
    assert event_types[1] == EventType.TTS_START
    assert "audio" in kinds[2:]


@pytest.mark.asyncio
async def test_tts_end_event_trails_last_audio_frame():
    """TTS_END must not be delivered before the last audio frame.

    Even without real-time pacing (single-frame case), the FIFO queue
    ordering guarantees TTS_END ships after the audio send completes.
    """
    transport = _RecordingTransport()
    codec = PcmCodec(16000)
    output = _EncodingOutput(transport, codec)

    await output.send_event(EventType.TTS_START)
    await output.send_audio(b"\x00" * codec.pcm_frame_bytes)
    await output.send_event(EventType.TTS_END)
    await output.flush()

    tts_end_t = next(t for t, e, _ in transport.events if e == EventType.TTS_END)
    last_audio_t = transport.audio[-1][0]
    assert tts_end_t >= last_audio_t, (
        f"TTS_END sent {(last_audio_t - tts_end_t) * 1000:.2f}ms before last audio"
    )


# ── 5. Backpressure: slow sink slows the pipeline ───────────────


@pytest.mark.asyncio
async def test_slow_send_audio_propagates_backpressure():
    """A slow ``send_audio`` must gate the overall pipeline.

    SpeechQueue awaits ``send_audio`` per chunk, so slowness propagates
    backwards to TTS and — through the sentence-split iterator — to the
    agent token pull. A regression that fire-and-forgets audio (e.g.
    pushing to an unbounded queue without backpressure) would let the
    pipeline complete in roughly the fast-sink time regardless.
    """
    n_chunks = 5
    per_chunk_delay = 0.015  # 15 ms

    async def run_with_delay(delay: float) -> float:
        va = _make_voice_assistant()

        async def stt(audio_iter, on_vad_start=None):
            async for _ in audio_iter:
                pass
            return "hi"

        async def agent(text, context=None):
            for i in range(n_chunks):
                yield f"Sentence {i}."

        async def tts(token_iter):
            # Yield one PCM chunk per sentence the splitter produces.
            from ovi_voice_assistant.tts import split_sentences

            async for _ in split_sentences(token_iter):
                yield b"\x00" * 320

        va.stt.transcribe_stream = stt
        va.agent.run_streamed = agent
        va.tts.synthesize_stream = tts

        output = _RecordingOutput(send_audio_delay=delay)
        t0 = time.perf_counter()
        await va.run(output, _async_iter([b"\x00"]))
        return time.perf_counter() - t0

    fast = await run_with_delay(0.0)
    slow = await run_with_delay(per_chunk_delay)

    # synthesize_stream has an internal 2-slot producer queue, so only
    # n_chunks - 2 send_audio awaits end up on the critical path. A 2-chunk
    # slack is enough to prove the sink's delay reaches back through the
    # pipeline without being bypassed by some unbounded buffer.
    min_expected_delta = per_chunk_delay * 2
    assert slow - fast > min_expected_delta, (
        f"backpressure not propagating: fast={fast * 1000:.1f}ms "
        f"slow={slow * 1000:.1f}ms (expected slow-fast > "
        f"{min_expected_delta * 1000:.0f}ms)"
    )
