"""WiFi transport — direct TCP connection to ESPHome device on port 6055."""

import asyncio
import logging
import socket
import struct

from ovi_voice_assistant.transport.device_transport import (
    AudioCallback,
    ConnectCallback,
    DeviceTransport,
    DisconnectCallback,
    EventCallback,
    EventType,
)

logger = logging.getLogger(__name__)

# TCP framing constants
MIC_AUDIO_TYPE = 0x20
SPEAKER_AUDIO_TYPE = 0x21
RECONNECT_DELAY = 3.0  # seconds between reconnect attempts


class WiFiTransport(DeviceTransport):
    """Transport over WiFi using a direct TCP connection.

    The Python server is the TCP client. The ESPHome device is the TCP
    server listening on port 6055. Communication uses a simple
    length-prefix binary protocol.

    TCP frame format:
        [2 bytes LE: payload length][payload bytes]

    Payload types:
        Control event:  [1 byte event_type][event payload]
        Mic audio:      [0x20][codec frame bytes]   (device -> server)
        Speaker audio:  [0x21][codec frame bytes]   (server -> device)
    """

    def __init__(
        self,
        host: str,
        port: int = 6055,
        encryption_key: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._encryption_key = encryption_key  # reserved for future Noise handshake

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._stopping = False
        self._recv_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Callbacks
        self._event_cb: EventCallback | None = None
        self._audio_cb: AudioCallback | None = None
        self._disconnect_cb: DisconnectCallback | None = None
        self._connect_cb: ConnectCallback | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- DeviceTransport interface --

    async def connect(self) -> None:
        """Connect to the device over TCP."""
        self._stopping = False
        await self._establish_connection()

    async def disconnect(self) -> None:
        """Disconnect from the device and stop reconnection."""
        self._stopping = True
        self._connected = False

        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reconnect_task = None

        await self._close_connection()

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        """Send a control event to the device."""
        if not self._connected:
            logger.warning("Cannot send event %s -- not connected", event.name)
            return
        self._send_frame(bytes([int(event)]) + payload)

    async def send_audio(self, data: bytes) -> None:
        """Send encoded audio to the device speaker."""
        if not self._connected:
            logger.warning("Cannot send audio -- not connected")
            return
        self._send_frame(bytes([SPEAKER_AUDIO_TYPE]) + data)

    def set_event_callback(self, callback: EventCallback) -> None:
        self._event_cb = callback

    def set_audio_callback(self, callback: AudioCallback) -> None:
        self._audio_cb = callback

    def set_disconnect_callback(self, callback: DisconnectCallback) -> None:
        self._disconnect_cb = callback

    def set_connect_callback(self, callback: ConnectCallback) -> None:
        self._connect_cb = callback

    # -- Internal: connection management --

    async def _establish_connection(self) -> None:
        """Open the TCP connection and start the receive loop."""
        logger.info("WiFi connecting to %s:%d", self._host, self._port)
        self._reader, self._writer = await asyncio.open_connection(
            self._host,
            self._port,
        )
        self._connected = True
        logger.info("WiFi connected to %s:%d", self._host, self._port)

        # Enable TCP keepalive so we detect a dead peer (e.g. device power
        # loss) within ~25s instead of waiting for the OS TCP timeout.
        sock: socket.socket = self._writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

        # TODO: Noise handshake when encryption_key is set

        # Start background receive loop
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _close_connection(self) -> None:
        """Close the TCP connection and cancel the receive loop."""
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None

        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    def _send_frame(self, data: bytes) -> None:
        """Send a length-prefixed frame over TCP."""
        if self._writer is None:
            return
        try:
            self._writer.write(struct.pack("<H", len(data)) + data)
        except Exception:
            logger.exception("Failed to send frame")

    # -- Internal: receive loop --

    async def _recv_loop(self) -> None:
        """Read length-prefixed frames from the device and dispatch."""
        if self._reader is None:
            return
        try:
            while self._connected:
                # Read 2-byte length prefix (little-endian)
                header = await self._reader.readexactly(2)
                length = struct.unpack("<H", header)[0]
                if length == 0:
                    continue

                payload = await self._reader.readexactly(length)
                msg_type = payload[0]
                msg_data = payload[1:]

                if msg_type == MIC_AUDIO_TYPE:
                    # Mic audio from device
                    if self._audio_cb is not None:
                        await self._audio_cb(msg_data)
                elif msg_type == SPEAKER_AUDIO_TYPE:
                    pass  # We don't receive speaker audio from device
                else:
                    # Control event
                    try:
                        event = EventType(msg_type)
                    except ValueError:
                        logger.debug("Unknown event type: 0x%02X", msg_type)
                        continue
                    if self._event_cb is not None:
                        await self._event_cb(event, msg_data)

        except asyncio.IncompleteReadError:
            logger.warning("WiFi connection closed by device (%s)", self._host)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("WiFi receive loop error (%s)", self._host)

        # Connection lost -- handle disconnect
        self._connected = False
        if self._disconnect_cb is not None:
            await self._disconnect_cb()

        if not self._stopping:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Retry connecting to the device after a disconnect."""
        while not self._stopping:
            await asyncio.sleep(RECONNECT_DELAY)
            try:
                logger.info("WiFi reconnecting to %s:%d ...", self._host, self._port)
                await self._close_connection()
                await self._establish_connection()
                logger.info("WiFi reconnected to %s:%d", self._host, self._port)
                if self._connect_cb is not None:
                    await self._connect_cb()
                return
            except Exception:
                logger.warning(
                    "WiFi reconnect to %s failed, retrying in %.0fs",
                    self._host,
                    RECONNECT_DELAY,
                )
