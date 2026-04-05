"""Manages connections to all configured Voice PE devices."""

import logging

from open_voice_assistant.config import DeviceConfig, Settings
from open_voice_assistant.device import DeviceConnection
from open_voice_assistant.voice_assistant_pipeline import VoiceAssistantPipeline

logger = logging.getLogger(__name__)


class DeviceManager:
    """Creates and manages DeviceConnection instances."""

    def __init__(self, devices: list[DeviceConfig], settings: Settings,
                 pipeline: VoiceAssistantPipeline) -> None:
        self._connections = [
            DeviceConnection(device, settings, pipeline)
            for device in devices
        ]

    async def start(self) -> None:
        for conn in self._connections:
            await conn.start()
        logger.info("Managing %d device(s)", len(self._connections))

    async def stop(self) -> None:
        for conn in self._connections:
            await conn.stop()
