"""Voice pipeline: STT → Agent → TTS with ESPHome device integration."""

import asyncio
import logging
from collections.abc import AsyncIterator

from aioesphomeapi import VoiceAssistantEventType
from aioesphomeapi.client import APIClient

from open_voice_assistant.agent.assistant import Assistant
from open_voice_assistant.agent.assistant_context import AssistantContext
from open_voice_assistant.config import Settings
from open_voice_assistant.stt import STT
from open_voice_assistant.tts import TTS

logger = logging.getLogger(__name__)

EVT = VoiceAssistantEventType
TTS_CHUNK_SIZE = 1024
LISTEN_TOKEN = "[LISTEN]"


STT_PROVIDERS: dict[str, type[STT]] = {}
TTS_PROVIDERS: dict[str, type[TTS]] = {}


def _create_stt(settings: Settings) -> STT:
    if settings.stt_provider not in STT_PROVIDERS:
        from open_voice_assistant.stt.whisper_stt import WhisperSTT
        STT_PROVIDERS["whisper"] = WhisperSTT
    cls = STT_PROVIDERS.get(settings.stt_provider)
    if cls is None:
        raise ValueError(f"Unknown STT provider: {settings.stt_provider}")
    return cls(settings)


def _create_tts(settings: Settings) -> TTS:
    if settings.tts_provider not in TTS_PROVIDERS:
        from open_voice_assistant.tts.piper_tts import PiperTTS
        TTS_PROVIDERS["piper"] = PiperTTS
        from open_voice_assistant.tts.pocket_tts import PocketTTS
        TTS_PROVIDERS["pocket"] = PocketTTS
    cls = TTS_PROVIDERS.get(settings.tts_provider)
    if cls is None:
        raise ValueError(f"Unknown TTS provider: {settings.tts_provider}")
    return cls(settings)


class VoiceAssistant:
    """Loads models and runs the full voice pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stt: STT = _create_stt(settings)
        self.tts: TTS = _create_tts(settings)
        self.agent = Assistant(settings)
        self._lock = asyncio.Lock()

    def load(self) -> None:
        self.stt.load()
        self.tts.load()
        self.agent.load()
        logger.info("Voice pipeline ready")

    async def start(self) -> None:
        """Start async resources (MCP servers, etc.)."""
        await self.agent.start()

    async def stop(self) -> None:
        """Stop async resources."""
        await self.agent.stop()

    def create_context(self, client: APIClient) -> AssistantContext:
        """Create an AssistantContext bound to a specific device connection."""
        return AssistantContext(
            announce=lambda text: asyncio.ensure_future(self.announce(client, text)),
        )

    async def run(self, client: APIClient, audio_iter: AsyncIterator[bytes],
                  context: AssistantContext | None = None) -> bool:
        """Run the full pipeline. Returns True if follow-up requested."""
        try:
            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_START, None)

            transcript = await self._run_stt(client, audio_iter)
            if not transcript:
                return False

            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_INTENT_START, None)
            agent_tokens = self.agent.run_streamed(transcript, context=context)

            needs_followup = await self._stream_tts(client, agent_tokens)

            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_INTENT_END, {
                "continue_conversation": "1" if needs_followup else "0",
            })
            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_END, None)
            logger.info("Pipeline complete (follow_up=%s)", needs_followup)
            return needs_followup

        except asyncio.CancelledError:
            logger.debug("Pipeline cancelled")
            return False
        except Exception:
            logger.exception("Pipeline error")
            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_ERROR,
                {"code": "pipeline_error", "message": "Pipeline processing failed"})
            return False

    async def _run_stt(self, client: APIClient, audio_iter: AsyncIterator[bytes]) -> str | None:
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_STT_START, None)

        async with self._lock:
            transcript = await self.stt.transcribe_stream(audio_iter)

        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_STT_VAD_END, None)

        if not transcript:
            logger.info("No speech detected")
            client.send_voice_assistant_event(
                EVT.VOICE_ASSISTANT_ERROR,
                {"code": "stt-no-text-recognized", "message": "No speech detected"},
            )
            client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_END, None)
            return None

        logger.debug("User said: %r", transcript)
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_STT_END, {"text": transcript})
        return transcript

    async def _stream_tts(self, client: APIClient, agent_tokens: AsyncIterator[str]) -> bool:
        """Stream TTS audio to device, paced to real-time. Returns follow-up flag."""
        # TTS_START starts speaker, TTS_END transitions to STREAMING_RESPONSE
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_START, {"text": "..."})
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_END, {"url": "api://audio"})
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_STREAM_START, None)

        follow_up = False
        full_text = ""

        async def filtered_tokens():
            nonlocal follow_up, full_text
            async for token in agent_tokens:
                full_text += token
                yield token
            if LISTEN_TOKEN in full_text:
                follow_up = True

        bps = self.tts.sample_rate * self.tts.sample_width * self.tts.channels
        total = 0
        t0 = 0.0
        header_sent = False
        loop = asyncio.get_running_loop()

        async for pcm_chunk in self.tts.synthesize_stream(filtered_tokens()):
            if not header_sent:
                header_sent = True
                t0 = loop.time()
                # # Prepend short silence so the device speaker is ready
                # silence = b"\x00" * (bps // 10)  # 100ms
                # for i in range(0, len(silence), TTS_CHUNK_SIZE):
                #     client.send_voice_assistant_audio(silence[i : i + TTS_CHUNK_SIZE])
                # total += len(silence)
                logger.info("TTS streaming started")

            for i in range(0, len(pcm_chunk), TTS_CHUNK_SIZE):
                chunk = pcm_chunk[i : i + TTS_CHUNK_SIZE]
                client.send_voice_assistant_audio(chunk)
                total += len(chunk)

                ahead = (total / bps) * 1.0 - (loop.time() - t0)
                if ahead > 0:
                    await asyncio.sleep(ahead)

        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_STREAM_END, None)

        if total == 0:
            logger.warning("TTS produced no audio")
        else:
            logger.info("TTS complete (%d bytes, %.1fs)", total, total / bps)

        return follow_up

    async def announce(self, client: APIClient, text: str) -> None:
        """Play a TTS announcement by driving a full voice assistant event sequence."""
        loop = asyncio.get_running_loop()
        audio: bytes = await loop.run_in_executor(None, self.tts.synthesize, text)
        if not audio:
            return

        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_START, None)
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_START, {"text": text})
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_END, {"url": "api://audio"})
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_STREAM_START, None)

        sr, sw, ch = self.tts.sample_rate, self.tts.sample_width, self.tts.channels
        bps = sr * sw * ch
        total = 0
        t0 = loop.time()
        for i in range(0, len(audio), TTS_CHUNK_SIZE):
            chunk = audio[i:i + TTS_CHUNK_SIZE]
            client.send_voice_assistant_audio(chunk)
            total += len(chunk)
            ahead = (total / bps) * 1.0 - (loop.time() - t0)
            if ahead > 0:
                await asyncio.sleep(ahead)

        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_TTS_STREAM_END, None)
        client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_END, None)
        logger.info("Announcement: %r (%.1fs)", text[:60], total / bps)
