"""ESPHome device connection — bridges Voice PE hardware to the voice pipeline."""

import asyncio
import logging
from collections.abc import AsyncIterator

from aioesphomeapi import VoiceAssistantAudioSettings
from aioesphomeapi.client import APIClient
from aioesphomeapi.reconnect_logic import ReconnectLogic

from open_voice_assistant.config import DeviceConfig, Settings
from open_voice_assistant.voice_assistant_pipeline import VoiceAssistantPipeline

logger = logging.getLogger(__name__)


class DeviceConnection:
    """Connects to a single Voice PE and routes audio through the pipeline."""

    def __init__(self, config: DeviceConfig, settings: Settings,
                 pipeline: VoiceAssistantPipeline) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._host = config.host
        self._port = config.port

        self._client = APIClient(
            address=config.host, port=config.port,
            password="", noise_psk=config.encryption_key or None,
        )
        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            zeroconf_instance=None,
        )
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._continuing = False

    async def start(self) -> None:
        await self._reconnect.start()
        logger.info("Connecting to %s:%d", self._host, self._port)

    async def stop(self) -> None:
        await self._cancel_task()
        await self._reconnect.stop()
        await self._client.disconnect()

    # -- Connection lifecycle --

    async def _on_connect(self) -> None:
        info = await self._client.device_info()
        logger.info("Connected: %s (ESPHome %s)", info.name, info.esphome_version)
        self._client.subscribe_voice_assistant(
            handle_start=self._on_session_start,
            handle_stop=self._on_session_stop,
            handle_audio=self._on_audio,
        )

    async def _on_disconnect(self, expected: bool) -> None:
        if not expected:
            logger.warning("Lost connection to %s, reconnecting", self._host)
        if self._task and not self._task.done():
            self._task.cancel()

    async def _on_session_start(self, conversation_id: str, flags: int,
                                audio_settings: VoiceAssistantAudioSettings,
                                wake_word_phrase: str | None) -> int | None:
        """Device detected wake word — start a new pipeline run."""
        await self._cancel_task()

        if self._continuing:
            logger.info("Follow-up session started")
            self._continuing = False
        else:
            await self._pipeline.agent.reset_history()
            logger.info("Voice session started (wake_word=%r)", wake_word_phrase)

        self._audio_queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())
        return 0  # use API_AUDIO (TCP), not UDP

    async def _on_session_stop(self, abort: bool) -> None:
        """Device ended audio stream (or abort requested)."""
        self._audio_queue.put_nowait(None)
        if abort:
            self._continuing = False
            if self._task and not self._task.done():
                self._task.cancel()

    async def _on_audio(self, data: bytes) -> None:
        """Device sent a chunk of mic audio."""
        self._audio_queue.put_nowait(data)

    async def _run(self) -> None:
        self._continuing = await self._pipeline.run(self._client, self._mic_stream())

    async def _mic_stream(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
