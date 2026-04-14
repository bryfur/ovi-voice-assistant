"""Unified device connection — bridges transport to the voice pipeline."""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from ovi_voice_assistant.agent.assistant_context import AssistantContext
from ovi_voice_assistant.codec import AudioCodec, create_codec
from ovi_voice_assistant.codec.lc3 import LC3_MUSIC_NBYTE
from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.music import MusicGroup, MusicPlayer
from ovi_voice_assistant.pipeline_output import PipelineOutput
from ovi_voice_assistant.transport import AudioConfig, DeviceTransport, EventType
from ovi_voice_assistant.voice_assistant import VoiceAssistant

if TYPE_CHECKING:
    from ovi_voice_assistant.memory import Memory
    from ovi_voice_assistant.scheduler import Scheduler

# Callback signature for wake-word arbitration.
# (connection, score, wake_word) -> None
WakeCallback = Callable[["DeviceConnection", int, str], Awaitable[None]]

logger = logging.getLogger(__name__)


class _EncodingOutput(PipelineOutput):
    """Adapts transport + codec to PipelineOutput.

    The pipeline sends raw PCM; this encodes it with the codec
    and sends the encoded frames paced to real-time, with a small
    lead time so the device buffer can absorb network jitter.
    """

    LEAD_TIME = 0.3  # seconds — send this much audio ahead of real-time

    def __init__(self, transport: DeviceTransport, codec: AudioCodec) -> None:
        self._transport = transport
        self._codec = codec
        self._pcm_buf = b""
        self._frame_count = 0
        self._t0 = 0.0
        self._frame_duration = codec.frame_duration_ms / 1000.0

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        # Send AUDIO_CONFIG before TTS_START so device configures decoder
        if event == EventType.TTS_START:
            config_payload = struct.pack(
                "<IHBB",
                self._codec.sample_rate,
                self._codec.encoded_frame_bytes,
                self._codec.codec_id,
                self._codec.channels,
            )
            await self._transport.send_event(EventType.AUDIO_CONFIG, config_payload)
        await self._transport.send_event(event, payload)

    async def send_audio(self, pcm: bytes) -> None:
        """Receive PCM from pipeline, encode with codec, pace to real-time."""
        loop = asyncio.get_running_loop()
        self._pcm_buf += pcm
        frame_bytes = self._codec.pcm_frame_bytes
        while len(self._pcm_buf) >= frame_bytes:
            frame = self._pcm_buf[:frame_bytes]
            self._pcm_buf = self._pcm_buf[frame_bytes:]
            encoded = self._codec.encode(frame)

            if self._frame_count == 0:
                self._t0 = loop.time()
            self._frame_count += 1

            await self._transport.send_audio(encoded)

            # Pace to real-time, but stay LEAD_TIME ahead so the device
            # buffer can absorb network jitter
            ahead = (self._frame_count * self._frame_duration) - (
                loop.time() - self._t0
            )
            if ahead > self.LEAD_TIME:
                await asyncio.sleep(ahead - self.LEAD_TIME)

    async def flush(self) -> None:
        """Flush any remaining PCM (pad to full frame)."""
        if self._pcm_buf:
            frame_bytes = self._codec.pcm_frame_bytes
            self._pcm_buf += b"\x00" * (frame_bytes - len(self._pcm_buf))
            encoded = self._codec.encode(self._pcm_buf)
            await self._transport.send_audio(encoded)
            self._pcm_buf = b""

    def reset(self) -> None:
        self._pcm_buf = b""
        self._frame_count = 0
        self._t0 = 0.0


class DeviceConnection:
    """Connects to a single Voice PE and routes audio through the pipeline."""

    def __init__(
        self,
        transport: DeviceTransport,
        spk_codec: AudioCodec,
        pipeline: VoiceAssistant,
        settings: Settings,
        on_wake: WakeCallback | None = None,
        name: str = "",
        music_group: MusicGroup | None = None,
    ) -> None:
        self._transport = transport
        self._spk_codec = spk_codec
        self._mic_codec: AudioCodec | None = (
            None  # created when device sends MIC_CONFIG
        )
        self._pipeline = pipeline
        self._settings = settings
        self._output = _EncodingOutput(transport, spk_codec)
        self._on_wake = on_wake
        self.name = name or repr(transport)

        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._continuing = False
        self._announcing = False
        self._context: AssistantContext | None = None
        self._music_output: _EncodingOutput | None = None
        self._music_group = music_group

    def set_scheduler(self, scheduler: Scheduler) -> None:
        """Attach the scheduler so automation tools are available."""
        if self._context:
            self._context.scheduler = scheduler

    def set_memory(self, memory: Memory) -> None:
        """Attach memory so remember/recall tools are available."""
        if self._context:
            self._context.memory = memory

    async def start(self) -> None:
        self._transport.set_event_callback(self._on_event)
        self._transport.set_audio_callback(self._on_audio)
        self._transport.set_disconnect_callback(self._on_disconnect)
        self._transport.set_connect_callback(self._on_reconnect)
        await self._transport.connect()
        await asyncio.sleep(0.5)
        await self._setup_device()

    async def stop(self) -> None:
        await self._cancel_task()
        await self._transport.disconnect()

    async def _setup_device(self) -> None:
        await self._transport.send_audio_config(
            AudioConfig(
                sample_rate=self._spk_codec.sample_rate,
                encoded_frame_bytes=self._spk_codec.encoded_frame_bytes,
                codec_type=self._spk_codec.codec_id,
                channels=self._spk_codec.channels,
            )
        )
        # Device responds with MIC_CONFIG after receiving AUDIO_CONFIG,
        # which configures self._mic_codec via _on_event.

        # Music gets its own 48 kHz codec so playback is CD-quality+
        # while TTS stays at whatever rate the voice codec uses.
        # AUDIO_CONFIG is re-sent before each TTS_START, so the device
        # reconfigures its decoder automatically.
        music_codec = create_codec(
            self._settings.transport.codec, 48000, channels=2, nbyte=LC3_MUSIC_NBYTE
        )
        self._music_output = _EncodingOutput(self._transport, music_codec)

        # Register this device's music output with the shared group
        if self._music_group is not None:
            self._music_group.add_device(self._music_output, self._transport)

        music_player = MusicPlayer(
            sample_rate=music_codec.sample_rate, channels=music_codec.channels
        )
        self._context = AssistantContext(
            announce=self.announce,
            music_player=music_player,
            music_group=self._music_group,
        )

    # -- Transport callbacks --

    async def _on_event(self, event: EventType, payload: bytes) -> None:
        if event == EventType.WAKE_WORD:
            await self._cancel_task()

            # Pause group music on ALL devices when any device wakes
            if self._music_group and self._music_group.is_active:
                await self._music_group.pause()

            if self._continuing:
                # Follow-up — already committed to this device, skip arbitration.
                logger.info("Follow-up session started")
                self._continuing = False
                self._audio_queue = asyncio.Queue()
                self._task = asyncio.create_task(self._run())
                return

            # Parse extended payload: [2B LE peak][2B LE ambient][wake_word UTF-8]
            # The normalised score (peak/ambient) cancels out mic gain
            # differences so the *closest* device wins, not the loudest mic.
            score = 0
            wake_word = ""
            if len(payload) >= 4:
                peak, ambient = struct.unpack_from("<HH", payload, 0)
                wake_word = payload[4:].decode("utf-8", errors="replace")
                score = peak * 1000 // max(ambient, 1)
                logger.debug(
                    "Wake payload: peak=%d ambient=%d → score=%d", peak, ambient, score
                )
            elif payload:
                wake_word = payload.decode("utf-8", errors="replace")

            # Prepare audio queue so mic frames arriving during
            # the arbitration window are buffered, not lost.
            self._audio_queue = asyncio.Queue()

            if self._on_wake is not None:
                await self._on_wake(self, score, wake_word)
            else:
                # No arbitration — start immediately.
                await self._pipeline.agent.reset_history()
                logger.info("Voice session started (wake_word=%r)", wake_word)
                self._task = asyncio.create_task(self._run())

        elif event == EventType.MIC_CONFIG:
            if len(payload) >= 7:
                rate = struct.unpack_from("<I", payload, 0)[0]
                nbyte = struct.unpack_from("<H", payload, 4)[0]
                codec_id = payload[6]
                codec_map = {0: "pcm", 1: "lc3", 2: "opus"}
                codec_name = codec_map.get(codec_id, "pcm")

                # Use the device's reported mic codec
                self._mic_codec = create_codec(codec_name, rate)
                logger.info(
                    "Device mic config: %s @ %dHz, %d bytes/frame",
                    codec_name,
                    rate,
                    nbyte,
                )

                # If the server prefers a different mic codec, request it
                preferred = self._settings.transport.codec
                if preferred != codec_name:
                    preferred_codec = create_codec(preferred, rate)
                    mic_config_payload = struct.pack(
                        "<IHB",
                        rate,
                        preferred_codec.encoded_frame_bytes,
                        preferred_codec.codec_id,
                    )
                    await self._transport.send_event(
                        EventType.MIC_CONFIG, mic_config_payload
                    )
                    self._mic_codec = preferred_codec
                    logger.info(
                        "Requested mic codec change: %s @ %dHz", preferred, rate
                    )

    async def _on_audio(self, data: bytes) -> None:
        """Decode mic audio and queue PCM for the pipeline."""
        if self._mic_codec is None:
            pcm = data
        else:
            pcm = self._mic_codec.decode(data)

        self._audio_queue.put_nowait(pcm)

    async def _on_disconnect(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _on_reconnect(self) -> None:
        await asyncio.sleep(0.5)
        await self._setup_device()

    # -- Pipeline execution --

    async def _run(self) -> None:
        self._output.reset()
        self._continuing = await self._pipeline.run(
            self._output,
            self._mic_stream(),
            context=self._context,
        )
        await self._output.flush()

        # After the voice pipeline completes, resume music if it was playing
        if not self._continuing and self._context:
            if self._music_group and self._music_group.is_active:
                # Group music was paused for voice — resume on all devices
                await self._music_group.resume()
            elif self._context.music_player and self._context.music_player.is_active:
                await self._run_music()

    async def _run_music(self) -> None:
        """Stream music at 48 kHz via the dedicated music output."""
        player = self._context.music_player
        out = self._music_output
        out.reset()
        await out.send_event(EventType.TTS_START)
        try:
            await player.stream(out)
            await out.flush()
            await out.send_event(EventType.TTS_END)
        except asyncio.CancelledError:
            try:
                await out.flush()
                await out.send_event(EventType.TTS_END)
            except Exception:
                pass
            raise

    async def _run_announce(self, text: str) -> None:
        try:
            self._output.reset()
            await self._pipeline.announce(self._output, text)
            await self._output.flush()
        finally:
            self._announcing = False

    # -- Helpers --

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
            except asyncio.CancelledError, Exception:
                pass

    async def start_pipeline(self, wake_word: str) -> None:
        """Called by DeviceManager when this device wins arbitration."""
        await self._pipeline.agent.reset_history()
        logger.info(
            "Voice session started on %s (wake_word=%r, won arbitration)",
            self.name,
            wake_word,
        )
        self._task = asyncio.create_task(self._run())

    async def abort_wake(self) -> None:
        """Called by DeviceManager when this device loses arbitration."""
        await self._cancel_task()
        # Drain any buffered mic audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._transport.send_event(EventType.WAKE_ABORT)
        logger.info("Wake aborted on %s (another device won)", self.name)

    def announce(self, text: str) -> asyncio.Task:
        """Stream a TTS announcement directly to the device."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.ensure_future(self._run_announce(text))
        return self._task
