"""Voice pipeline: STT → Agent → TTS with event-driven output."""

import asyncio
import logging
from collections.abc import AsyncIterator

from ovi_voice_assistant.agent.assistant import Assistant
from ovi_voice_assistant.agent.assistant_context import AssistantContext
from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.speech_queue import SpeechQueue
from ovi_voice_assistant.stt import STT
from ovi_voice_assistant.transport import EventType
from ovi_voice_assistant.tts import TTS

logger = logging.getLogger(__name__)

LISTEN_TOKEN = "[LISTEN]"

STT_PROVIDERS: dict[str, type[STT]] = {}
TTS_PROVIDERS: dict[str, type[TTS]] = {}


def _create_stt(settings: Settings) -> STT:
    if settings.stt.provider not in STT_PROVIDERS:
        if settings.stt.provider == "whisper":
            from ovi_voice_assistant.stt.whisper_stt import WhisperSTT

            STT_PROVIDERS["whisper"] = WhisperSTT
        elif settings.stt.provider == "nemotron":
            from ovi_voice_assistant.stt.nemotron_stt import NemotronSTT

            STT_PROVIDERS["nemotron"] = NemotronSTT
    cls = STT_PROVIDERS.get(settings.stt.provider)
    if cls is None:
        raise ValueError(f"Unknown STT provider: {settings.stt.provider}")
    return cls(settings)


def _create_tts(settings: Settings, sample_rate: int = 16000) -> TTS:
    if settings.tts.provider not in TTS_PROVIDERS:
        if settings.tts.provider == "piper":
            from ovi_voice_assistant.tts.piper_tts import PiperTTS

            TTS_PROVIDERS["piper"] = PiperTTS
        elif settings.tts.provider == "kokoro":
            from ovi_voice_assistant.tts.kokoro_tts import KokoroTTS

            TTS_PROVIDERS["kokoro"] = KokoroTTS
        elif settings.tts.provider == "qwen3":
            from ovi_voice_assistant.tts.qwen3_tts import Qwen3TTS

            TTS_PROVIDERS["qwen3"] = Qwen3TTS
    cls = TTS_PROVIDERS.get(settings.tts.provider)
    if cls is None:
        raise ValueError(f"Unknown TTS provider: {settings.tts.provider}")
    return cls(settings, sample_rate=sample_rate)


class VoiceAssistant:
    """Loads models and runs the full voice pipeline (STT → Agent → TTS).

    Works entirely in PCM — codec encoding/decoding is handled by the caller.
    """

    def __init__(self, settings: Settings, tts_sample_rate: int = 16000) -> None:
        self.settings = settings
        self.stt: STT = _create_stt(settings)
        self.tts: TTS = _create_tts(settings, sample_rate=tts_sample_rate)
        self.agent = Assistant(settings)
        self._last_response = ""
        self._bg_tasks: set[asyncio.Task] = set()

    def load(self) -> None:
        self.stt.load()
        self.tts.load()
        self.agent.load()
        logger.info("Voice pipeline ready")

    async def start(self) -> None:
        await self.agent.start()

    async def stop(self) -> None:
        await self.agent.stop()

    async def run(
        self,
        output: PipelineOutput,
        audio_iter: AsyncIterator[bytes],
        context: AssistantContext | None = None,
    ) -> bool:
        """Run the full pipeline. Returns True if follow-up requested."""
        speech = SpeechQueue(self.tts, output)
        try:
            transcript = await self._run_stt(output, audio_iter)
            if not transcript:
                return False

            if context:
                context.say = self._make_say_callback(speech)

            agent_tokens = self.agent.run_streamed(transcript, context=context)
            needs_followup = await self._stream_tts(output, speech, agent_tokens)

            if needs_followup:
                await output.send_event(EventType.CONTINUE)

            # Auto-retain: fire-and-forget memory extraction
            if context and context.memory:
                exchange = f"User: {transcript}\nAssistant: {self._last_response}"
                task = asyncio.create_task(
                    self._retain_memory(context.memory, exchange),
                    name="memory-retain",
                )
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

            logger.info("Pipeline complete (follow_up=%s)", needs_followup)
            return needs_followup

        except asyncio.CancelledError:
            logger.debug("Pipeline cancelled")
            return False
        except Exception:
            logger.exception("Pipeline error")
            try:
                await output.send_event(
                    EventType.ERROR, b"pipeline_error\0Pipeline processing failed"
                )
            except Exception:
                pass
            return False
        finally:
            await speech.stop()

    async def announce(self, output: PipelineOutput, text: str) -> None:
        """Play a TTS announcement."""
        speech = SpeechQueue(self.tts, output)
        try:
            await output.send_event(EventType.TTS_START)
            await speech.submit(text)
            await output.send_event(EventType.TTS_END)
            logger.info("Announcement complete: %r", text[:60])
        except asyncio.CancelledError:
            logger.debug("Announcement cancelled")
        except Exception:
            logger.exception("Announcement error")
        finally:
            await speech.stop()

    def _make_say_callback(self, speech: SpeechQueue):
        """Create a callback that enqueues text for playback and returns at once.

        The SpeechQueue serializes say utterances with response TTS, so the
        agent's say call is fire-and-forget without risk of overlapping the
        main response audio on the device.
        """

        async def _say(text: str) -> None:
            speech.submit(text)

        return _say

    # -- Internal --

    async def _run_stt(
        self, output: PipelineOutput, audio_iter: AsyncIterator[bytes]
    ) -> str | None:
        async def on_vad_start():
            await output.send_event(EventType.VAD_START)

        transcript = await self.stt.transcribe_stream(
            audio_iter, on_vad_start=on_vad_start
        )

        await output.send_event(EventType.MIC_STOP)

        if not transcript:
            logger.info("No speech detected")
            await output.send_event(EventType.ERROR, b"stt-no-text\0No speech detected")
            return None

        logger.debug("User said: %r", transcript)
        return transcript

    async def _stream_tts(
        self,
        output: PipelineOutput,
        speech: SpeechQueue,
        agent_tokens: AsyncIterator[str],
    ) -> bool:
        follow_up = False
        full_text = ""

        async def filtered_tokens():
            nonlocal follow_up, full_text
            async for token in agent_tokens:
                full_text += token
                yield token
            if LISTEN_TOKEN in full_text:
                follow_up = True

        await output.send_event(EventType.TTS_START)
        await speech.submit_stream(filtered_tokens())
        await output.send_event(EventType.TTS_END)
        logger.info("Speaking complete: %r", full_text[:80])
        self._last_response = full_text
        return follow_up

    @staticmethod
    async def _retain_memory(memory, exchange: str) -> None:
        """Background task to extract and store facts from a conversation turn."""
        try:
            await memory.retain(exchange, context="voice conversation")
        except Exception:
            logger.exception("Background memory retain failed")
