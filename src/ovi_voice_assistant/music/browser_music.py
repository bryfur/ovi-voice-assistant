"""Base class for browser-based music streaming via tab audio capture.

Launches Chromium (Playwright), navigates to a music service, captures tab
audio with getDisplayMedia, and streams s16le PCM over a local WebSocket
back to Python.  Subclasses provide service-specific search / play logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ovi_voice_assistant.music.music_player import MusicTrack
from ovi_voice_assistant.pipeline_output import PipelineOutput

logger = logging.getLogger(__name__)

_PROFILE_ROOT = Path("~/.config/ovi")

_CAPTURE_JS = """\
async (wsPort) => {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      suppressLocalAudioPlayback: true,
    },
    video: true,
    preferCurrentTab: true,
  });
  stream.getVideoTracks().forEach(t => t.stop());

  const audioTrack = stream.getAudioTracks()[0];
  const settings = audioTrack?.getSettings?.() || {};
  const channels = settings.channelCount || 2;
  console.log('[ovi] audio track:', audioTrack?.label,
              'channels:', channels, 'sampleRate:', settings.sampleRate);

  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, channels, channels);

  const ws = new WebSocket(`ws://127.0.0.1:${wsPort}`);
  ws.binaryType = 'arraybuffer';
  await new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', () => rej(new Error('WebSocket connection failed')));
  });

  ws.send(new TextEncoder().encode(JSON.stringify({
    sampleRate: ctx.sampleRate,
    channels: channels,
  })));

  proc.onaudioprocess = (e) => {
    const len = e.inputBuffer.getChannelData(0).length;
    const pcm = new Int16Array(len * channels);
    for (let ch = 0; ch < channels; ch++) {
      const data = e.inputBuffer.getChannelData(ch);
      for (let i = 0; i < len; i++)
        pcm[i * channels + ch] = Math.max(-32768, Math.min(32767, data[i] * 32768));
    }
    if (ws.readyState === 1) ws.send(pcm.buffer);
  };
  source.connect(proc);
  proc.connect(ctx.destination);
  window.__oviCapture = { ctx, ws, stream };
}
"""


class BrowserMusic:
    """Base class for streaming music via browser tab audio capture.

    Subclasses must set ``_URL`` and ``_PROFILE_NAME`` and implement
    :meth:`search`, :meth:`_play_and_wait`, and :meth:`stop_playback`.
    """

    _URL: str
    _PROFILE_NAME: str

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._pw = None
        self._context = None
        self._page = None
        self._ws_server = None
        self._ws_port: int = 0
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._capture_active = False
        self._browser_sample_rate: int = 48000
        self._browser_channels: int = 2

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch browser, open the music service, start the WebSocket bridge."""
        import websockets
        from playwright.async_api import async_playwright

        self._ws_server = await websockets.serve(self._ws_handler, "127.0.0.1", 0)
        self._ws_port = self._ws_server.sockets[0].getsockname()[1]
        logger.info("Audio bridge listening on ws://127.0.0.1:%d", self._ws_port)

        self._pw = await async_playwright().start()
        profile = (_PROFILE_ROOT / self._PROFILE_NAME).expanduser()
        profile.mkdir(parents=True, exist_ok=True)

        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            headless=False,
            args=[
                "--auto-accept-this-tab-capture",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        self._page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )

        cdp = await self._context.new_cdp_session(self._page)
        await cdp.send("Page.setBypassCSP", {"enabled": True})

        await self._page.goto(self._URL, wait_until="domcontentloaded")

    async def close(self) -> None:
        """Shut down browser and WebSocket server."""
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    # ── interface (subclasses implement) ─────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[MusicTrack]:
        raise NotImplementedError

    async def _play_and_wait(self, track: MusicTrack) -> None:
        """Start playback and return when the track finishes."""
        raise NotImplementedError

    async def stop_playback(self) -> None:
        raise NotImplementedError

    # ── streaming (shared) ───────────────────────────────────────────────

    async def stream_track(self, track: MusicTrack, output: PipelineOutput) -> None:
        """Play *track* in the browser and forward captured PCM to *output*."""
        await self._ensure_capture()

        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

        play_done = asyncio.create_task(self._play_and_wait(track))

        while not play_done.done():
            try:
                data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            if data is None:
                break
            await output.send_audio(data)

        while not self._audio_queue.empty():
            data = self._audio_queue.get_nowait()
            if data:
                await output.send_audio(data)

    # ── internals ────────────────────────────────────────────────────────

    async def _ws_handler(self, ws) -> None:
        logger.info("Browser audio capture connected")
        first = True
        try:
            async for msg in ws:
                if first and isinstance(msg, str):
                    cfg = json.loads(msg)
                    self._browser_sample_rate = cfg.get("sampleRate", 48000)
                    self._browser_channels = cfg.get("channels", 2)
                    logger.info(
                        "Browser audio: %dHz %dch",
                        self._browser_sample_rate,
                        self._browser_channels,
                    )
                    first = False
                    continue
                first = False
                if isinstance(msg, bytes):
                    await self._audio_queue.put(msg)
        except Exception:
            logger.exception("Audio bridge error")
        finally:
            await self._audio_queue.put(None)

    async def _ensure_capture(self) -> None:
        if self._capture_active:
            return
        await self._page.evaluate(_CAPTURE_JS, self._ws_port)
        self._capture_active = True
        logger.info("Tab audio capture active")
