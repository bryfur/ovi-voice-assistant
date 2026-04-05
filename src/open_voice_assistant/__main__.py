"""Entry point for the voice assistant server."""

import asyncio
import argparse
import logging
import signal
import sys

from open_voice_assistant.config import DeviceConfig, Settings, parse_devices
from open_voice_assistant.device_manager import DeviceManager
from open_voice_assistant.voice_assistant_pipeline import VoiceAssistantPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Voice Assistant")
    parser.add_argument(
        "devices",
        nargs="*",
        help="Device addresses: IP, hostname.local, or host:port:encryption_key",
    )
    parser.add_argument("--scan", action="store_true", help="Scan for ESPHome devices on the network")
    parser.add_argument("--gen-key", action="store_true", help="Generate an encryption key and exit")
    parser.add_argument("--stt-provider", default=None, help="STT provider (whisper)")
    parser.add_argument("--tts-provider", default=None, help="TTS provider (piper)")
    parser.add_argument("--stt-model", default=None, help="STT model name")
    parser.add_argument("--tts-model", default=None, help="TTS model name")
    parser.add_argument("--agent-model", default=None, help="OpenAI model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("open_voice_assistant")

    if args.gen_key:
        import base64, secrets
        from pathlib import Path

        key = base64.b64encode(secrets.token_bytes(32)).decode()

        # Write to esphome/secrets.yaml
        secrets_path = Path("esphome/secrets.yaml")
        if secrets_path.exists():
            content = secrets_path.read_text()
            if "api_encryption_key" not in content:
                content += f"api_encryption_key: \"{key}\"\n"
                secrets_path.write_text(content)
                print(f"Added api_encryption_key to {secrets_path}")
            else:
                print(f"api_encryption_key already exists in {secrets_path}")
                print(f"New key (not saved): {key}")
        else:
            print(f"{secrets_path} not found — key not saved")
            print(f"Key: {key}")

        print(f"\nEnable in esphome/voice-pe.yaml:")
        print(f"  api:")
        print(f"    id: api_id")
        print(f"    encryption:")
        print(f"      key: !secret api_encryption_key")
        print(f"\nConnect with:")
        print(f"  ova DEVICE_HOST:6053::{key}")
        return

    # Scan mode — just discover and print devices
    if args.scan:
        asyncio.run(_scan())
        return

    # Build settings from env + CLI
    overrides = {}
    if args.stt_provider:
        overrides["stt_provider"] = args.stt_provider
    if args.tts_provider:
        overrides["tts_provider"] = args.tts_provider
    if args.stt_model:
        overrides["stt_model"] = args.stt_model
    if args.tts_model:
        overrides["tts_model"] = args.tts_model
    if args.agent_model:
        overrides["agent_model"] = args.agent_model

    settings = Settings(**overrides)

    # Parse device addresses from CLI args
    # CLI devices override env devices
    if args.devices:
        devices = parse_devices(",".join(args.devices))
    else:
        devices = settings.get_devices()

    if not devices:
        logger.error(
            "No devices configured. Usage:\n"
            "  ova voice-pe-XXXX.local           # connect by mDNS hostname\n"
            "  ova 192.168.1.100                  # connect by IP\n"
            "  ova --scan                         # discover devices on network\n"
            "  ova 192.168.1.100:6053:KEY          # with encryption key\n"
        )
        sys.exit(1)

    logger.info("Starting Open Voice Assistant")
    logger.info("  STT: %s (%s)", settings.stt_provider, settings.stt_model)
    logger.info("  TTS: %s (%s)", settings.tts_provider, settings.tts_model)
    logger.info("  Agent: %s", settings.agent_model)
    any_encrypted = False
    for i, d in enumerate(devices):
        logger.info("  Device %d: %s:%d", i, d.host, d.port)
        if d.encryption_key:
            any_encrypted = True

    if not any_encrypted:
        logger.warning(
            "No encryption configured. Communication is unencrypted. "
            "Run 'ova --gen-key' to generate a key and enable encryption."
        )

    # Load models (synchronous, done before serving)
    pipeline = VoiceAssistantPipeline(settings)
    pipeline.load()

    # Run async server
    asyncio.run(_serve(devices, settings, pipeline))


async def _scan() -> None:
    """Discover ESPHome devices on the local network."""
    from open_voice_assistant.discovery import discover_devices

    print("Scanning for ESPHome devices (5s)...")
    devices = await discover_devices(timeout=5.0)

    if not devices:
        print("No devices found. Make sure your Voice PE is powered on and connected to WiFi.")
        return

    print(f"\nFound {len(devices)} device(s):\n")
    print(f"  {'Name':<30} {'IP':<18} {'Port'}")
    print(f"  {'─' * 30} {'─' * 18} {'─' * 5}")
    for d in devices:
        print(f"  {d['name']:<30} {d['ip']:<18} {d['port']}")

    print(f"\nConnect with:")
    for d in devices:
        print(f"  ova {d['host']}")


async def _serve(devices: list[DeviceConfig], settings: Settings,
                 pipeline: VoiceAssistantPipeline) -> None:
    server = DeviceManager(devices, settings, pipeline)
    await server.start()

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    logging.getLogger("open_voice_assistant").info("Shutting down...")
    await server.stop()


if __name__ == "__main__":
    main()
