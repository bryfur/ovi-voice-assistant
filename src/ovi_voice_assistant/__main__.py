"""Entry point for the voice assistant server."""

import argparse
import asyncio
import logging
import signal
import sys

from ovi_voice_assistant.config import DeviceConfig, Settings, parse_devices
from ovi_voice_assistant.voice_assistant import VoiceAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Ovi — Open Voice Assistant")
    parser.add_argument(
        "devices",
        nargs="*",
        help="Device addresses: IP, hostname.local, or host:port:encryption_key (default port: 6055)",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Scan for ESPHome devices on the network"
    )
    parser.add_argument(
        "--gen-key", action="store_true", help="Generate an encryption key and exit"
    )
    parser.add_argument("--stt-provider", default=None, help="STT provider (whisper, nemotron)")
    parser.add_argument(
        "--tts-provider", default=None, help="TTS provider (piper, pocket)"
    )
    parser.add_argument("--stt-model", default=None, help="STT model name")
    parser.add_argument("--tts-model", default=None, help="TTS model name")
    parser.add_argument("--agent-model", default=None, help="OpenAI model name")
    parser.add_argument(
        "--mcp-servers",
        default=None,
        help='MCP servers JSON: \'[{"command": "npx", "args": ["-y", "@dangahagan/weather-mcp"]}]\'',
    )
    parser.add_argument(
        "--agents", default=None, help="Sub-agents JSON or @path/to/agents.json"
    )
    parser.add_argument(
        "--transport",
        default=None,
        choices=["wifi", "ble"],
        help="Transport: wifi or ble (default: wifi)",
    )
    parser.add_argument(
        "--codec",
        default=None,
        choices=["pcm", "lc3", "opus"],
        help="Audio codec: pcm, lc3, opus (default: lc3)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers even in debug mode
    for name in ("aioesphomeapi", "httpx", "httpcore", "openai", "bleak", "pocket_tts"):
        logging.getLogger(name).setLevel(logging.INFO)
    logger = logging.getLogger("ovi_voice_assistant")

    if args.gen_key:
        import base64
        import secrets
        from pathlib import Path

        key = base64.b64encode(secrets.token_bytes(32)).decode()

        # Write to esphome/secrets.yaml
        secrets_path = Path("esphome/secrets.yaml")
        if secrets_path.exists():
            content = secrets_path.read_text()
            if "api_encryption_key" not in content:
                content += f'api_encryption_key: "{key}"\n'
                secrets_path.write_text(content)
                print(f"Added api_encryption_key to {secrets_path}")
            else:
                print(f"api_encryption_key already exists in {secrets_path}")
                print(f"New key (not saved): {key}")
        else:
            print(f"{secrets_path} not found — key not saved")
            print(f"Key: {key}")

        print("\nConnect with:")
        print(f"  ovi DEVICE_HOST:6055::{key}")
        return

    # Scan mode — just discover and print devices
    if args.scan:
        asyncio.run(_scan())
        return

    # Build settings from env + CLI
    cli_fields = [
        "stt_provider",
        "tts_provider",
        "stt_model",
        "tts_model",
        "agent_model",
        "mcp_servers",
        "agents",
        "transport",
        "codec",
    ]
    overrides = {
        f: getattr(args, f) for f in cli_fields if getattr(args, f) is not None
    }

    settings = Settings(**overrides)

    # Determine TTS sample rate.
    # First create a temporary codec to find the actual rate it'll use
    # (e.g., LC3 snaps 22050→24000), then create TTS at that rate.
    from ovi_voice_assistant.codec import create_codec

    probe_rate = settings.speaker_sample_rate if settings.speaker_sample_rate else 24000
    probe_codec = create_codec(settings.codec, probe_rate)
    tts_sample_rate = probe_codec.sample_rate  # actual codec rate

    logger.info("Starting Ovi")
    logger.info("  Transport: %s", settings.transport)
    logger.info("  Codec: %s", settings.codec)
    logger.info("  STT: %s (%s)", settings.stt_provider, settings.stt_model)
    logger.info("  TTS: %s (%s)", settings.tts_provider, settings.tts_model)
    logger.info("  Agent: %s", settings.agent_model)
    if settings.mcp_servers:
        logger.info("  MCP: %s", settings.mcp_servers)

    # Initialize memory if enabled
    memory = None
    if settings.memory_enabled:
        from ovi_voice_assistant.memory import Memory

        memory = Memory(settings)
        memory.load()
        logger.info("  Memory: SQLite (bank=%s)", settings.memory_bank_id)

    # Load models — TTS uses the codec's actual sample rate
    pipeline = VoiceAssistant(settings, tts_sample_rate=tts_sample_rate)
    pipeline.load()

    actual_tts_rate = pipeline.tts.sample_rate

    if settings.transport == "ble":
        # BLE mode — single device, direct connection
        logger.info("  Speaker sample rate: %d Hz", actual_tts_rate)
        asyncio.run(_serve_ble(settings, pipeline, actual_tts_rate, memory=memory))
    else:
        # WiFi mode — one or more devices via DeviceManager
        devices = _resolve_devices(args, settings, logger)
        if devices is None:
            sys.exit(1)
        logger.info("  Speaker sample rate: %d Hz", actual_tts_rate)
        asyncio.run(
            _serve_wifi(devices, settings, pipeline, actual_tts_rate, memory=memory)
        )


def _resolve_devices(args, settings, logger) -> list[DeviceConfig] | None:
    """Parse device addresses from CLI args or settings."""
    if args.devices:
        devices = parse_devices(",".join(args.devices))
    else:
        devices = settings.get_devices()

    if not devices:
        logger.error(
            "No devices configured. Usage:\n"
            "  ovi voice-pe-XXXX.local           # connect by mDNS hostname\n"
            "  ovi 192.168.1.100                  # connect by IP\n"
            "  ovi --scan                         # discover devices on network\n"
            "  ovi 192.168.1.100:6055:KEY          # with encryption key\n"
            "  ovi --transport ble                 # connect over Bluetooth\n"
        )
        return None

    any_encrypted = False
    for i, d in enumerate(devices):
        logger.info("  Device %d: %s:%d", i, d.host, d.port)
        if d.encryption_key:
            any_encrypted = True

    if not any_encrypted:
        logger.warning(
            "No encryption configured. Communication is unencrypted. "
            "Run 'ovi --gen-key' to generate a key and enable encryption."
        )

    return devices


async def _scan() -> None:
    """Discover ESPHome devices on the local network."""
    from ovi_voice_assistant.discovery import discover_devices

    print("Scanning for ESPHome devices (5s)...")
    devices = await discover_devices(timeout=5.0)

    if not devices:
        print(
            "No devices found. Make sure your Voice PE is powered on and connected to WiFi."
        )
        return

    print(f"\nFound {len(devices)} device(s):\n")
    print(f"  {'Name':<30} {'IP':<18} {'Port'}")
    print(f"  {'─' * 30} {'─' * 18} {'─' * 5}")
    for d in devices:
        print(f"  {d['name']:<30} {d['ip']:<18} {d['port']}")

    print("\nConnect with:")
    for d in devices:
        print(f"  ovi {d['host']}")


def _create_scheduler(
    settings: Settings,
    pipeline: VoiceAssistant,
    announce_fn,
):
    """Create and load the scheduler for proactive automations."""
    from pathlib import Path

    from ovi_voice_assistant.scheduler import Scheduler

    path = Path(settings.automations_path).expanduser()
    scheduler = Scheduler(
        path=path,
        run_prompt=pipeline.agent.run_text,
        announce=announce_fn,
    )
    scheduler.load()
    return scheduler


async def _serve_wifi(
    devices: list[DeviceConfig],
    settings: Settings,
    pipeline: VoiceAssistant,
    tts_rate: int,
    memory=None,
) -> None:
    """Serve WiFi devices via DeviceManager."""
    from ovi_voice_assistant.device_manager import DeviceManager

    await pipeline.start()
    server = DeviceManager(devices, settings, pipeline, tts_rate)
    await server.start()

    scheduler = _create_scheduler(settings, pipeline, server.announce_all)
    server.set_scheduler(scheduler)
    if memory:
        server.set_memory(memory)
    await scheduler.start()

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    logging.getLogger("ovi_voice_assistant").info("Shutting down...")
    await scheduler.stop()
    try:
        await asyncio.wait_for(server.stop(), timeout=5.0)
    except TimeoutError:
        pass
    try:
        await asyncio.wait_for(pipeline.stop(), timeout=5.0)
    except TimeoutError:
        pass
    if memory:
        memory.close()


async def _serve_ble(
    settings: Settings,
    pipeline: VoiceAssistant,
    tts_rate: int,
    memory=None,
) -> None:
    """Serve a single BLE device."""
    from ovi_voice_assistant.codec import create_codec
    from ovi_voice_assistant.device_connection import DeviceConnection
    from ovi_voice_assistant.transport.ble import BLETransport

    transport = BLETransport(settings.ble_device_name, settings.ble_device_address)
    codec = create_codec(settings.codec, tts_rate)

    await pipeline.start()
    conn = DeviceConnection(transport, codec, pipeline, settings)
    await conn.start()

    async def announce_ble(text: str) -> None:
        conn.announce(text)

    scheduler = _create_scheduler(settings, pipeline, announce_ble)
    conn.set_scheduler(scheduler)
    if memory:
        conn.set_memory(memory)
    await scheduler.start()

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    logging.getLogger("ovi_voice_assistant").info("Shutting down...")
    await scheduler.stop()
    try:
        await asyncio.wait_for(conn.stop(), timeout=5.0)
    except TimeoutError:
        pass
    try:
        await asyncio.wait_for(pipeline.stop(), timeout=5.0)
    except TimeoutError:
        pass
    if memory:
        memory.close()


if __name__ == "__main__":
    main()
