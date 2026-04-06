"""ESPHome device connection — bridges Voice PE hardware to the voice pipeline."""

import asyncio
import logging
from collections.abc import AsyncIterator

from aioesphomeapi import ButtonInfo, VoiceAssistantAudioSettings, VoiceAssistantEventType as EVT
from aioesphomeapi.client import APIClient
from aioesphomeapi.reconnect_logic import ReconnectLogic

from open_voice_assistant.config import DeviceConfig, Settings
from open_voice_assistant.voice_assistant import VoiceAssistant

logger = logging.getLogger(__name__)

ANNOUNCE_BUTTON_ID = "announce"


class DeviceConnection:
    """Connects to a single Voice PE and routes audio through the pipeline."""

    def __init__(self, config: DeviceConfig, settings: Settings,
                 pipeline: VoiceAssistant) -> None:
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
        self._context = None
        self._announce_button_key: int | None = None
        self._pending_announcement: str | None = None
        self._announcing = False

    async def start(self) -> None:
        await self._reconnect.start()
        logger.info("Connecting to %s:%d", self._host, self._port)

    async def stop(self) -> None:
        await self._cancel_task()
        await self._reconnect.stop()
        await self._client.disconnect()

    # -- Connection lifecycle --

    async def _on_connect(self) -> None:
        device_info, entities, _ = await self._client.device_info_and_list_entities()
        logger.info("Connected: %s (ESPHome %s)", device_info.name, device_info.esphome_version)

        for entity in entities:
            if isinstance(entity, ButtonInfo):
                logger.info("Button entity: object_id=%r name=%r key=%d", entity.object_id, entity.name, entity.key)
            if isinstance(entity, ButtonInfo) and entity.object_id == ANNOUNCE_BUTTON_ID:
                self._announce_button_key = entity.key
                logger.info("Found announce button (key=%d)", entity.key)
                break

        self._context = self._pipeline.create_context(self._client)
        if self._announce_button_key is not None:
            self._context.announce = self._announce
        else:
            logger.warning(
                "No '%s' button entity found — announcements (timers) disabled. "
                "Add the announce button to your ESPHome config.",
                ANNOUNCE_BUTTON_ID,
            )

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
        """Device started a session — either wake word or server-triggered announcement."""
        await self._cancel_task()

        if self._announcing:
            logger.debug("Skipping post-announcement session")
            self._announcing = False
            self._client.send_voice_assistant_event(EVT.VOICE_ASSISTANT_RUN_END, None)
            return 0

        if self._pending_announcement is not None:
            text = self._pending_announcement
            self._pending_announcement = None
            logger.info("Announcement session started")
            self._announcing = True
            self._task = asyncio.create_task(self._run_announce(text))
        else:
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
        self._continuing = await self._pipeline.run(
            self._client, self._mic_stream(), context=self._context,
        )

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

    async def _run_announce(self, text: str) -> None:
        try:
            await self._pipeline.announce(self._client, text)
        finally:
            self._announcing = False

    # -- Server-initiated announcements --

    def _announce(self, text: str) -> asyncio.Task:
        """Trigger a TTS announcement by starting a device voice session.

        Sets pending text and presses the announce button on the device,
        which triggers voice_assistant.start → allocates speaker buffers →
        calls handle_start → we detect the pending announcement and stream
        TTS audio through the normal voice assistant audio path.
        """
        self._pending_announcement = text
        self._client.button_command(self._announce_button_key)
        return asyncio.ensure_future(asyncio.sleep(0))
