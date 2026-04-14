"""BLE transport — wraps bleak for GATT-based communication with voice devices."""

import asyncio
import logging
import struct

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

from ovi_voice_assistant.transport.device_transport import (
    AudioCallback,
    ConnectCallback,
    DeviceTransport,
    DisconnectCallback,
    EventCallback,
    EventType,
)

logger = logging.getLogger(__name__)

# GATT UUIDs for the Ovi voice service
SERVICE_UUID = "BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0"
AUDIO_TX_UUID = "BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0"  # mic → server (notify)
AUDIO_RX_UUID = (
    "BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0"  # server → speaker (write-no-resp)
)
CONTROL_UUID = "BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0"  # events (read/write/notify)

# BLE ATT MTU minus overhead — safe default for write-without-response
DEFAULT_MTU_PAYLOAD = 512

# Reconnection parameters
RECONNECT_DELAY = 2.0  # seconds between reconnect attempts
SCAN_TIMEOUT = 10.0  # seconds to scan for device


class BLETransport(DeviceTransport):
    """Transport over Bluetooth Low Energy using bleak."""

    def __init__(
        self,
        device_name: str | None = None,
        device_address: str | None = None,
        mtu_payload: int = DEFAULT_MTU_PAYLOAD,
        auto_reconnect: bool = True,
    ) -> None:
        """Initialize BLE transport.

        Provide either device_name (scan for it) or device_address (connect directly).
        """
        if not device_name and not device_address:
            raise ValueError("Must provide device_name or device_address")
        self._device_name = device_name
        self._device_address = device_address
        self._mtu_payload = mtu_payload
        self._auto_reconnect = auto_reconnect

        self._ble_device: BLEDevice | None = None
        self._bleak_client: BleakClient | None = None
        self._connected = False
        self._reconnect_task: asyncio.Task | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Callbacks
        self._event_cb: EventCallback | None = None
        self._audio_cb: AudioCallback | None = None
        self._disconnect_cb: DisconnectCallback | None = None
        self._connect_cb: ConnectCallback | None = None

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._bleak_client is not None
            and self._bleak_client.is_connected
        )

    # -- DeviceTransport interface --

    async def connect(self) -> None:
        """Scan for and connect to the BLE device."""
        self._stopping = False

        if self._ble_device is None:
            self._ble_device = await self._scan()

        await self._establish_connection()

    async def disconnect(self) -> None:
        """Disconnect from the BLE device and stop reconnection."""
        self._stopping = True
        self._connected = False

        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reconnect_task = None

        if self._bleak_client is not None and self._bleak_client.is_connected:
            try:
                await self._bleak_client.stop_notify(AUDIO_TX_UUID)
            except Exception:
                pass
            try:
                await self._bleak_client.stop_notify(CONTROL_UUID)
            except Exception:
                pass
            await self._bleak_client.disconnect()
            logger.info("BLE transport disconnected")

        self._bleak_client = None

    async def send_event(self, event: EventType, payload: bytes = b"") -> None:
        """Send a control event to the device via the CONTROL characteristic.

        BLE event protocol: [1 byte event_type][payload]
        """
        if not self.is_connected:
            logger.warning("Cannot send event %s — not connected", event.name)
            return

        data = struct.pack("B", int(event)) + payload
        try:
            await self._bleak_client.write_gatt_char(
                CONTROL_UUID,
                data,
                response=False,
            )
        except Exception:
            logger.exception("Failed to send event %s", event.name)

    async def send_audio(self, data: bytes) -> None:
        """Send audio data to the device speaker via the AUDIO_RX characteristic.

        Audio is chunked to fit within the BLE MTU.
        """
        if not self.is_connected:
            logger.warning("Cannot send audio — not connected")
            return

        # Chunk audio to fit within MTU
        for offset in range(0, len(data), self._mtu_payload):
            chunk = data[offset : offset + self._mtu_payload]
            try:
                await self._bleak_client.write_gatt_char(
                    AUDIO_RX_UUID,
                    chunk,
                    response=False,
                )
            except Exception:
                logger.exception("Failed to send audio chunk")
                return

    def set_event_callback(self, callback: EventCallback) -> None:
        self._event_cb = callback

    def set_audio_callback(self, callback: AudioCallback) -> None:
        self._audio_cb = callback

    def set_disconnect_callback(self, callback: DisconnectCallback) -> None:
        self._disconnect_cb = callback

    def set_connect_callback(self, callback: ConnectCallback) -> None:
        self._connect_cb = callback

    # -- Internal: scanning --

    async def _scan(self) -> BLEDevice:
        """Scan for the target BLE device by name or address."""
        logger.info(
            "Scanning for BLE device (name=%r, address=%r) ...",
            self._device_name,
            self._device_address,
        )

        if self._device_address:
            device = await BleakScanner.find_device_by_address(
                self._device_address,
                timeout=SCAN_TIMEOUT,
            )
            if device is None:
                raise ConnectionError(
                    f"BLE device with address {self._device_address} not found"
                )
            logger.info("Found BLE device: %s (%s)", device.name, device.address)
            return device

        # Scan by name — look for our service UUID
        devices = await BleakScanner.discover(
            timeout=SCAN_TIMEOUT,
            service_uuids=[SERVICE_UUID.lower()],
        )
        for device in devices:
            if (
                device.name
                and self._device_name
                and self._device_name.lower() in device.name.lower()
            ):
                logger.info("Found BLE device: %s (%s)", device.name, device.address)
                return device

        # Fallback: any device advertising our service UUID
        if devices:
            device = devices[0]
            logger.info(
                "Found BLE device by service UUID: %s (%s)", device.name, device.address
            )
            return device

        raise ConnectionError(
            f"No BLE device found (name={self._device_name!r}, service={SERVICE_UUID})"
        )

    # -- Internal: connection --

    async def _establish_connection(self) -> None:
        """Connect to the BLE device and subscribe to GATT notifications."""
        if self._ble_device is None:
            raise RuntimeError("No BLE device — call connect() first")
        self._loop = asyncio.get_running_loop()

        self._bleak_client = BleakClient(
            self._ble_device,
            disconnected_callback=self._on_bleak_disconnect,
        )

        logger.info(
            "Connecting to BLE device: %s (%s)",
            self._ble_device.name,
            self._ble_device.address,
        )
        await self._bleak_client.connect()
        self._connected = True
        logger.info(
            "BLE connected to %s", self._ble_device.name or self._ble_device.address
        )

        # Subscribe to audio notifications (mic → server)
        await self._bleak_client.start_notify(AUDIO_TX_UUID, self._on_audio_notify)
        logger.debug("Subscribed to AUDIO_TX notifications")

        # Subscribe to control notifications (device events → server)
        await self._bleak_client.start_notify(CONTROL_UUID, self._on_control_notify)
        logger.debug("Subscribed to CONTROL notifications")

    def _on_bleak_disconnect(self, client: BleakClient) -> None:
        """Called by bleak when the BLE connection drops."""
        was_connected = self._connected
        self._connected = False

        if self._stopping:
            return

        if was_connected:
            logger.warning(
                "BLE disconnected from %s",
                self._ble_device.name if self._ble_device else "unknown",
            )

        # Fire disconnect callback and schedule reconnect
        if self._loop is not None and self._disconnect_cb is not None:
            self._loop.create_task(self._disconnect_cb())

        if self._loop is not None and self._auto_reconnect and not self._stopping:
            self._reconnect_task = self._loop.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect to the BLE device after disconnection."""
        while not self._stopping:
            await asyncio.sleep(RECONNECT_DELAY)
            try:
                logger.info(
                    "BLE reconnecting to %s ...",
                    self._ble_device.name if self._ble_device else "unknown",
                )
                # Re-scan in case the device address changed (e.g. random MAC)
                try:
                    self._ble_device = await self._scan()
                except ConnectionError:
                    logger.warning(
                        "BLE device not found during reconnect scan, retrying"
                    )
                    continue

                await self._establish_connection()
                logger.info("BLE reconnected successfully")
                if self._connect_cb is not None:
                    await self._connect_cb()
                return
            except Exception:
                logger.warning(
                    "BLE reconnect failed, retrying in %.0fs", RECONNECT_DELAY
                )

    # -- Internal: GATT notification handlers --

    def _on_audio_notify(self, _sender: int, data: bytearray) -> None:
        """Handle incoming audio data from the device microphone (AUDIO_TX)."""
        if self._audio_cb is not None and self._loop is not None:
            self._loop.create_task(self._audio_cb(bytes(data)))

    def _on_control_notify(self, _sender: int, data: bytearray) -> None:
        """Handle incoming control events from the device (CONTROL).

        BLE event protocol: [1 byte event_type][payload]
        """
        if len(data) < 1:
            logger.warning("Received empty control notification")
            return

        try:
            event_type = EventType(data[0])
        except ValueError:
            logger.warning("Unknown BLE event type: 0x%02X", data[0])
            return

        payload = bytes(data[1:])

        if self._event_cb is not None and self._loop is not None:
            self._loop.create_task(self._event_cb(event_type, payload))
