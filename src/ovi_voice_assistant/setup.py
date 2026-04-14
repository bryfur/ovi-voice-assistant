"""Interactive CLI setup wizard for first-run configuration."""

import asyncio
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ovi_voice_assistant.config import CONFIG_PATH

console = Console()

# Available options for each provider
STT_PROVIDERS = ["nemotron", "whisper"]
TTS_PROVIDERS = ["kokoro", "piper"]
NEMOTRON_MODELS = {
    "int8-dynamic": "best for CPU (recommended)",
    "fp16": "best for NVIDIA GPU",
    "int8-static": "GPU, less VRAM than fp16",
    "fp32": "original precision, largest",
}
WHISPER_MODELS = {
    "tiny.en": "fastest, least accurate (English)",
    "tiny": "fastest, least accurate (multilingual)",
    "base.en": "good balance (English, recommended)",
    "base": "good balance (multilingual)",
    "small.en": "more accurate, slower (English)",
    "small": "more accurate, slower (multilingual)",
    "medium.en": "high accuracy, slow (English)",
    "medium": "high accuracy, slow (multilingual)",
    "large-v3": "best accuracy, slowest",
    "turbo": "large-v3 speed-optimized (recommended for GPU)",
    "distil-large-v3.5": "distilled, fast + accurate (English)",
    "distil-large-v3": "distilled, fast + accurate (English)",
    "distil-medium.en": "distilled medium (English)",
    "distil-small.en": "distilled small (English)",
}
KOKORO_VOICES = {
    "af_heart": "Female American (default)",
    "af_bella": "Female American",
    "af_nicole": "Female American",
    "af_sarah": "Female American",
    "af_sky": "Female American",
    "am_adam": "Male American",
    "am_michael": "Male American",
    "bf_emma": "Female British",
    "bf_isabella": "Female British",
    "bm_george": "Male British",
    "bm_lewis": "Male British",
}
CODECS = {
    "lc3": "low latency, good quality (recommended)",
    "opus": "high quality, higher latency",
    "pcm": "uncompressed, highest bandwidth",
}


def _pick(label: str, options: dict[str, str], default: str) -> str:
    """Show numbered options with descriptions and let the user pick one."""
    items = list(options.items())
    console.print(f"\n  [bold]{label}[/bold]")
    for i, (key, desc) in enumerate(items, 1):
        marker = " [dim](default)[/dim]" if key == default else ""
        console.print(f"    [cyan]{i}.[/cyan] {key} — [dim]{desc}[/dim]{marker}")

    while True:
        raw = click.prompt("  Choice", default="", show_default=False)
        if not raw:
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1][0]
        except ValueError:
            if raw in options:
                return raw
        console.print(f"  [red]Enter a number 1-{len(items)}[/red]")


def _load_existing() -> dict:
    """Load existing config as a nested dict. Returns empty dict if no file."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _get(existing: dict, *keys: str, default: str = "") -> str:
    """Get a nested value from the existing config dict."""
    node = existing
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node if node is not None else default


def _scan_devices() -> list[dict]:
    """Run device discovery and return results."""
    try:
        from ovi_voice_assistant.discovery import discover_devices

        with console.status("Scanning for ESPHome devices..."):
            devices = asyncio.run(discover_devices(timeout=5.0))
        return devices or []
    except Exception as e:
        console.print(f"  [red]Scan failed:[/red] {e}")
        return []


def _select_scanned_devices(existing: list[str]) -> list[str]:
    """Scan for devices and let the user pick which to add."""
    found = _scan_devices()
    if not found:
        console.print("  [yellow]No devices found on the network.[/yellow]")
        return existing

    console.print(f"\n  Found [bold]{len(found)}[/bold] device(s):")
    for i, d in enumerate(found, 1):
        already = " [dim](already configured)[/dim]" if d["host"] in existing else ""
        console.print(
            f"    [cyan]{i}.[/cyan] {d['name']} ({d['ip']}:{d['port']}){already}"
        )

    sel = click.prompt(
        "\n  Add devices (comma-separated numbers, 'all', or 'none')",
        default="all",
    )
    if sel.lower() == "none":
        return existing

    if sel.lower() == "all":
        new_hosts = [d["host"] for d in found]
    else:
        indices = [int(x.strip()) - 1 for x in sel.split(",") if x.strip()]
        new_hosts = [found[i]["host"] for i in indices if 0 <= i < len(found)]

    # Merge with existing, dedup
    return list(dict.fromkeys(existing + new_hosts))


def run_setup() -> dict:
    """Run the interactive setup wizard.

    If a config file already exists, loads it and uses current values as
    defaults so the user can press Enter to keep existing settings.
    Returns the nested config dict.
    """
    existing = _load_existing()
    is_edit = bool(existing)

    console.print()
    if is_edit:
        console.print(
            Panel(
                "[bold]Ovi — Open Voice Assistant[/bold]\n"
                "Edit configuration — press Enter to keep current values",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[bold]Ovi — Open Voice Assistant[/bold]\nFirst-time setup wizard",
                expand=False,
            )
        )
        console.print("Press Enter to accept defaults. Type a number to select.\n")

    config: dict = {}

    # ── LLM ──────────────────────────────────────────────────
    console.rule("[bold]LLM[/bold]")
    console.print(
        "  Ovi uses an LLM for conversation. You can use OpenAI\n"
        "  or any compatible API (Ollama, LM Studio, etc).\n"
    )

    llm: dict = {}
    llm["api_key"] = click.prompt(
        "  API key",
        default=_get(existing, "llm", "api_key"),
        hide_input=True,
        show_default=False,
    )
    llm["base_url"] = click.prompt(
        "  Base URL (empty for OpenAI)",
        default=_get(existing, "llm", "base_url"),
        show_default=False,
    )
    llm["model"] = click.prompt(
        "  Model",
        default=_get(existing, "llm", "model", default="gpt-4o-mini"),
    )
    config["llm"] = {k: v for k, v in llm.items() if v}

    # ── STT ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Speech-to-Text[/bold]")
    console.print(
        "  [bold]Nemotron[/bold] — NVIDIA Nemotron Speech 600M, streaming RNNT (recommended).\n"
        "  [bold]Whisper[/bold] — OpenAI Whisper via faster-whisper, multilingual.\n"
    )

    stt: dict = {}
    stt["provider"] = click.prompt(
        "  Provider",
        type=click.Choice(STT_PROVIDERS, case_sensitive=False),
        default=_get(existing, "stt", "provider", default="nemotron"),
    )

    if stt["provider"] == "nemotron":
        stt["model"] = _pick(
            "Nemotron variant:",
            NEMOTRON_MODELS,
            default=_get(existing, "stt", "model", default="int8-dynamic"),
        )
    else:
        stt["model"] = _pick(
            "Whisper model:",
            WHISPER_MODELS,
            default=_get(existing, "stt", "model", default="base.en"),
        )

    stt["device"] = click.prompt(
        "  Compute device",
        type=click.Choice(["cpu", "cuda"], case_sensitive=False),
        default=_get(existing, "stt", "device", default="cpu"),
    )
    config["stt"] = stt

    # ── TTS ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Text-to-Speech[/bold]")
    console.print(
        "  [bold]Kokoro[/bold] — fast, high-quality local TTS.\n"
        "  [bold]Piper[/bold] — lighter-weight, lower quality.\n"
    )

    tts: dict = {}
    tts["provider"] = click.prompt(
        "  Provider",
        type=click.Choice(TTS_PROVIDERS, case_sensitive=False),
        default=_get(existing, "tts", "provider", default="kokoro"),
    )

    if tts["provider"] == "kokoro":
        tts["model"] = _pick(
            "Kokoro voice:",
            KOKORO_VOICES,
            default=_get(existing, "tts", "model", default="af_heart"),
        )
    else:
        tts["model"] = click.prompt(
            "  Piper voice model",
            default=_get(existing, "tts", "model", default="en_US-lessac-medium"),
        )
    config["tts"] = tts

    # ── Devices ──────────────────────────────────────────────
    console.print()
    console.rule("[bold]Devices[/bold]")
    console.print("  Connect to ESPHome devices running the Ovi component.\n")

    existing_devices_raw = _get(existing, "devices")
    if isinstance(existing_devices_raw, list):
        current_devices = existing_devices_raw
    elif isinstance(existing_devices_raw, str) and existing_devices_raw:
        current_devices = [
            d.strip() for d in existing_devices_raw.split(",") if d.strip()
        ]
    else:
        current_devices = []

    if current_devices:
        console.print("  Current devices:")
        for d in current_devices:
            console.print(f"    [green]•[/green] {d}")
        console.print()

    if click.confirm("  Scan for devices on the network?", default=True):
        device_list = _select_scanned_devices(current_devices)
        if device_list:
            console.print(f"  Devices: [green]{', '.join(device_list)}[/green]")
    elif current_devices:
        raw = click.prompt(
            "  Devices (comma-separated, or Enter to keep current)",
            default=", ".join(current_devices),
        )
        device_list = [d.strip() for d in raw.split(",") if d.strip()]
    else:
        raw = click.prompt(
            "  Device address (IP or hostname.local, comma-separated)",
            default="",
            show_default=False,
        )
        device_list = [d.strip() for d in raw.split(",") if d.strip()]

    if device_list:
        config["devices"] = device_list

    # ── Transport ────────────────────────────────────────────
    console.print()
    console.rule("[bold]Transport[/bold]")

    transport: dict = {}
    transport["codec"] = _pick(
        "Audio codec:",
        CODECS,
        default=_get(existing, "transport", "codec", default="lc3"),
    )
    config["transport"] = transport

    # ── Summary & Save ───────────────────────────────────────
    console.print()
    table = Table(title="Configuration Summary", show_header=False, expand=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                display = "****" if key == "api_key" else str(value)
                table.add_row(f"{section}.{key}", display)
        elif isinstance(values, list):
            table.add_row(section, ", ".join(str(v) for v in values))
        else:
            table.add_row(section, str(values))
    console.print(table)

    console.print()
    if click.confirm("  Save configuration?", default=True):
        save_config(config)
        console.print(f"\n  [green]Configuration saved to {CONFIG_PATH}[/green]")
        console.print("  Edit the file directly or re-run: [bold]ovi --setup[/bold]")
        console.print(
            "  Override any value with env vars: [bold]OVI_LLM__MODEL=gpt-4o[/bold]"
        )
    else:
        console.print("\n  [yellow]Configuration not saved.[/yellow]")

    return config


def save_config(config: dict, path: Path | None = None) -> None:
    """Write nested config dict as YAML."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Ovi configuration", "# Edit directly or run: ovi --setup", ""]

    for key, value in config.items():
        lines.append(yaml.dump({key: value}, default_flow_style=False).strip())
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def needs_setup() -> bool:
    """Check if first-run setup is needed (no config file exists)."""
    return not CONFIG_PATH.exists()
