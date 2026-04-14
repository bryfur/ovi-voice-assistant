"""Run context passed to agent tools for device interaction."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovi_voice_assistant.memory import Memory
    from ovi_voice_assistant.music import MusicGroup, MusicPlayer
    from ovi_voice_assistant.scheduler import Scheduler

logger = logging.getLogger(__name__)


@dataclass
class AssistantContext:
    """Context available to all agent tools during a pipeline run.

    Created by ``VoiceAssistant.create_context`` — one per device connection.
    The ``announce`` callback drives a full voice assistant event sequence
    so the device activates its speaker and plays the audio.
    """

    announce: Callable[[str], asyncio.Task] | None = None
    """Trigger a TTS announcement on the device. Returns the background task."""

    say: Callable[[str], Awaitable[None]] | None = None
    """Speak text immediately during a pipeline run without interrupting it."""

    music_player: MusicPlayer | None = None
    """Music player for this device — controls queue, playback state."""

    music_group: MusicGroup | None = None
    """Shared music group for synchronized multi-device playback."""

    scheduler: Scheduler | None = None
    """Scheduler for creating/managing proactive automations."""

    memory: Memory | None = None
    """Persistent memory for fact extraction and recall across conversations."""

    _timers: dict[str, tuple[asyncio.TimerHandle, float]] = field(default_factory=dict)
    """Maps label → (handle, fire_time) where fire_time is loop.time() epoch."""

    _pending_say: asyncio.Task | None = field(default=None, init=False, repr=False)
    """Background task for in-flight say() audio."""

    async def drain_say(self) -> None:
        """Wait for any in-flight say() audio to finish."""
        task = self._pending_say
        self._pending_say = None
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                logger.debug("say() task failed", exc_info=True)

    def schedule_timer(self, seconds: float, label: str) -> None:
        """Schedule a timer that announces on the device when it fires."""
        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            seconds,
            lambda: loop.create_task(self._on_timer_fire(label)),
        )
        self._timers[label] = (handle, loop.time() + seconds)
        logger.info("Timer scheduled: %s (%.0fs)", label, seconds)

    def cancel_timer(self, label: str) -> bool:
        """Cancel a named timer. Returns True if found."""
        entry = self._timers.pop(label, None)
        if entry:
            entry[0].cancel()
            logger.info("Timer cancelled: %s", label)
            return True
        return False

    def get_timer_status(self) -> dict[str, float]:
        """Return remaining seconds for each active timer."""
        now = asyncio.get_running_loop().time()
        return {
            label: max(0.0, fire_time - now)
            for label, (_, fire_time) in self._timers.items()
        }

    async def _on_timer_fire(self, label: str) -> None:
        """Announce timer completion on the device."""
        self._timers.pop(label, None)
        text = f"Your {label} timer is done."
        logger.info("Timer fired: %s", label)
        if self.announce:
            self.announce(text)
        else:
            logger.warning("No announce callback — cannot notify device")
