"""mDNS discovery for ESPHome Voice PE devices on the local network."""

import asyncio
import logging

from zeroconf import ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

logger = logging.getLogger(__name__)

ESPHOME_SERVICE_TYPE = "_esphomelib._tcp.local."


async def discover_devices(timeout: float = 5.0) -> list[dict[str, str | int]]:
    """Scan the local network for ESPHome devices via mDNS.

    Returns a list of dicts with keys: name, host, ip, port.
    """
    devices: list[dict[str, str | int]] = []
    found_names: set[str] = set()

    def on_service_state_change(
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change == ServiceStateChange.Added:
            found_names.add(name)

    aiozc = AsyncZeroconf()
    browser = AsyncServiceBrowser(
        aiozc.zeroconf, ESPHOME_SERVICE_TYPE, handlers=[on_service_state_change]
    )

    # Wait for responses
    await asyncio.sleep(timeout)

    # Resolve all found services
    for name in found_names:
        info = AsyncServiceInfo(ESPHOME_SERVICE_TYPE, name)
        await info.async_request(aiozc.zeroconf, 3000)
        if info.parsed_addresses():
            ip = info.parsed_addresses()[0]
            port = info.port or 6055
            friendly_name = name.replace(f".{ESPHOME_SERVICE_TYPE}", "")
            devices.append(
                {
                    "name": friendly_name,
                    "host": f"{friendly_name}.local",
                    "ip": ip,
                    "port": port,
                }
            )

    await browser.async_cancel()
    await aiozc.async_close()

    return devices
