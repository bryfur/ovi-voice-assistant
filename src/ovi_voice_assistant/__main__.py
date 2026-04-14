"""Entry point for the voice assistant server."""

import asyncio
import logging
import signal
import sys

import rich_click as click

from ovi_voice_assistant.config import DeviceConfig, Settings, parse_devices
from ovi_voice_assistant.voice_assistant import VoiceAssistant

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True


@click.command()
@click.argument("devices", nargs=-1)
@click.option("--scan", is_flag=True, help="Scan for ESPHome devices on the network.")
@click.option("--gen-key", is_flag=True, help="Generate an encryption key and exit.")
@click.option("--setup", is_flag=True, help="Run the interactive setup wizard.")
@click.option("--flash", is_flag=True, help="Flash ESPHome firmware to a device.")
@click.option("--stt-provider", default=None, help="STT provider (whisper).")
@click.option("--tts-provider", default=None, help="TTS provider (kokoro, piper).")
@click.option("--stt-model", default=None, help="STT model name.")
@click.option("--tts-model", default=None, help="TTS model name.")
@click.option("--agent-model", default=None, help="LLM model name.")
@click.option(
    "--mcp-servers",
    default=None,
    help='MCP servers JSON: \'[{"command": "npx", "args": [...]}]\'.',
)
@click.option("--agents", default=None, help="Sub-agents JSON or @path/to/agents.json.")
@click.option(
    "--transport",
    default=None,
    type=click.Choice(["wifi", "ble"], case_sensitive=False),
    help="Transport: wifi or ble.",
)
@click.option(
    "--codec",
    default=None,
    type=click.Choice(["pcm", "lc3", "opus"], case_sensitive=False),
    help="Audio codec: pcm, lc3, opus.",
)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def main(
    devices,
    scan,
    gen_key,
    setup,
    flash,
    stt_provider,
    tts_provider,
    stt_model,
    tts_model,
    agent_model,
    mcp_servers,
    agents,
    transport,
    codec,
    debug,
) -> None:
    """Ovi — Open Voice Assistant.

    Connect to ESPHome devices by passing DEVICES as IP addresses,
    hostnames, or host:port:key triples.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers even in debug mode
    for name in ("aioesphomeapi", "httpx", "httpcore", "openai", "bleak"):
        logging.getLogger(name).setLevel(logging.INFO)
    logger = logging.getLogger("ovi_voice_assistant")

    if gen_key:
        _gen_key()
        return

    # Scan mode — just discover and print devices
    if scan:
        asyncio.run(_scan())
        return

    # Flash mode — compile and upload ESPHome firmware
    if flash:
        from ovi_voice_assistant.flash import run_flash

        run_flash()
        return

    # Setup wizard — run explicitly or on first start
    from ovi_voice_assistant.setup import needs_setup, run_setup

    if setup:
        run_setup()
        return

    if needs_setup() and sys.stdin.isatty():
        click.echo("No configuration found. Running first-time setup...\n")
        run_setup()
        click.echo()

    # Build overrides dict — CLI args override nested config via init kwargs
    overrides: dict = {}
    if stt_provider:
        overrides.setdefault("stt", {})["provider"] = stt_provider
    if stt_model:
        overrides.setdefault("stt", {})["model"] = stt_model
    if tts_provider:
        overrides.setdefault("tts", {})["provider"] = tts_provider
    if tts_model:
        overrides.setdefault("tts", {})["model"] = tts_model
    if agent_model:
        overrides.setdefault("llm", {})["model"] = agent_model
    if mcp_servers:
        overrides.setdefault("llm", {})["mcp_servers"] = mcp_servers
    if agents:
        overrides.setdefault("llm", {})["agents"] = agents
    if transport:
        overrides.setdefault("transport", {})["type"] = transport
    if codec:
        overrides.setdefault("transport", {})["codec"] = codec

    settings = Settings(**overrides)

    # Determine TTS sample rate.
    # First create a temporary codec to find the actual rate it'll use
    # (e.g., LC3 snaps 22050→24000), then create TTS at that rate.
    from ovi_voice_assistant.codec import create_codec

    probe_rate = (
        settings.transport.speaker_sample_rate
        if settings.transport.speaker_sample_rate
        else 24000
    )
    probe_codec = create_codec(settings.transport.codec, probe_rate)
    tts_sample_rate = probe_codec.sample_rate  # actual codec rate

    logger.info("Starting Ovi")
    logger.info("  Transport: %s", settings.transport.type)
    logger.info("  Codec: %s", settings.transport.codec)
    logger.info("  STT: %s (%s)", settings.stt.provider, settings.stt.model)
    logger.info("  TTS: %s (%s)", settings.tts.provider, settings.tts.model)
    logger.info("  Agent: %s", settings.llm.model)
    if settings.llm.mcp_servers:
        logger.info("  MCP: %s", settings.llm.mcp_servers)

    # Initialize memory if enabled
    memory = None
    if settings.memory.enabled:
        from ovi_voice_assistant.memory import Memory

        memory = Memory(settings)
        memory.load()
        logger.info("  Memory: SQLite (bank=%s)", settings.memory.bank_id)

    # Load models — TTS uses the codec's actual sample rate
    pipeline = VoiceAssistant(settings, tts_sample_rate=tts_sample_rate)
    pipeline.load()

    actual_tts_rate = pipeline.tts.sample_rate

    if settings.transport.type == "ble":
        # BLE mode — single device, direct connection
        logger.info("  Speaker sample rate: %d Hz", actual_tts_rate)
        asyncio.run(_serve_ble(settings, pipeline, actual_tts_rate, memory=memory))
    else:
        # WiFi mode — one or more devices via DeviceManager
        resolved = _resolve_devices(devices, settings, logger)
        if resolved is None:
            sys.exit(1)
        logger.info("  Speaker sample rate: %d Hz", actual_tts_rate)
        asyncio.run(
            _serve_wifi(resolved, settings, pipeline, actual_tts_rate, memory=memory)
        )


def _gen_key() -> None:
    """Generate an encryption key and optionally write to esphome/secrets.yaml."""
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
            click.echo(f"Added api_encryption_key to {secrets_path}")
        else:
            click.echo(f"api_encryption_key already exists in {secrets_path}")
            click.echo(f"New key (not saved): {key}")
    else:
        click.echo(f"{secrets_path} not found — key not saved")
        click.echo(f"Key: {key}")

    click.echo("\nConnect with:")
    click.echo(f"  ovi DEVICE_HOST:6055::{key}")


def _resolve_devices(
    cli_devices: tuple, settings: Settings, logger
) -> list[DeviceConfig] | None:
    """Parse device addresses from CLI args or settings."""
    if cli_devices:
        devices = parse_devices(",".join(cli_devices))
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
    import yaml

    from ovi_voice_assistant.config import CONFIG_PATH
    from ovi_voice_assistant.discovery import discover_devices

    click.echo("Scanning for ESPHome devices (5s)...")
    devices = await discover_devices(timeout=5.0)

    if not devices:
        click.echo(
            "No devices found. Make sure your Voice PE is powered on and connected to WiFi."
        )
        return

    # Load existing config to show which devices are already added
    existing_hosts: list[str] = []
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("devices", [])
        if isinstance(raw, list):
            existing_hosts = raw
        elif isinstance(raw, str):
            existing_hosts = [d.strip() for d in raw.split(",") if d.strip()]

    click.echo(f"\nFound {len(devices)} device(s):\n")
    for i, d in enumerate(devices, 1):
        tag = " (configured)" if d["host"] in existing_hosts else ""
        click.echo(f"  {i}. {d['name']:<28} {d['ip']:<18} {d['port']}{tag}")

    # Offer to add devices to config
    new_devices = [d for d in devices if d["host"] not in existing_hosts]
    if new_devices and sys.stdin.isatty():
        click.echo()
        sel = click.prompt(
            "Add devices to config (comma-separated numbers, 'all', or Enter to skip)",
            default="",
            show_default=False,
        )
        if sel.strip():
            if sel.strip().lower() == "all":
                hosts_to_add = [d["host"] for d in new_devices]
            else:
                indices = []
                for s in sel.split(","):
                    try:
                        idx = int(s.strip()) - 1
                        if (
                            0 <= idx < len(devices)
                            and devices[idx]["host"] not in existing_hosts
                        ):
                            indices.append(idx)
                    except ValueError:
                        pass
                hosts_to_add = [devices[i]["host"] for i in indices]

            if hosts_to_add:
                from ovi_voice_assistant.setup import add_devices_to_config

                add_devices_to_config(hosts_to_add)
                for h in hosts_to_add:
                    click.echo(f"  Added {h}")
                click.echo(f"\nConfiguration saved to {CONFIG_PATH}")
                return

    click.echo("\nConnect with:")
    for d in devices:
        click.echo(f"  ovi {d['host']}")


def _create_scheduler(
    settings: Settings,
    pipeline: VoiceAssistant,
    announce_fn,
):
    """Create and load the scheduler for proactive automations."""
    from pathlib import Path

    from ovi_voice_assistant.scheduler import Scheduler

    path = Path(settings.automations.path).expanduser()
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

    transport = BLETransport(settings.ble.device_name, settings.ble.device_address)
    codec = create_codec(settings.transport.codec, tts_rate)

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
