"""Interactive device flashing via ESPHome."""

import asyncio
import platform
import re
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

ESPHOME_DIR = Path("esphome")


# Preferred ordering for device configs (first match wins top position)
_PREFERRED_ORDER = ["voice-pe", "atom-echo", "s3-box-3"]


def _find_device_configs() -> list[dict]:
    """Find ESPHome device YAML configs and extract display names."""
    configs = []
    for path in sorted(ESPHOME_DIR.glob("*.yaml")):
        if path.name == "secrets.yaml":
            continue
        # Extract description from first comment line
        first_line = path.read_text().split("\n", 1)[0]
        match = re.search(r"config for (.+)$", first_line)
        if match:
            description = match.group(1)
        else:
            # Try BLE variant pattern
            match = re.search(r"— (.+) for (.+)$", first_line)
            description = f"{match.group(2)} ({match.group(1)})" if match else path.stem
        configs.append({"path": path, "name": path.stem, "description": description})

    # Sort preferred devices to the top
    def sort_key(cfg):
        name = cfg["name"]
        for i, prefix in enumerate(_PREFERRED_ORDER):
            if name.startswith(prefix):
                return (i, name)
        return (len(_PREFERRED_ORDER), name)

    configs.sort(key=sort_key)
    return configs


def _check_secrets() -> bool:
    """Check if secrets.yaml has real WiFi credentials and an encryption key."""
    secrets_path = ESPHOME_DIR / "secrets.yaml"
    if not secrets_path.exists():
        return False
    content = secrets_path.read_text()
    has_wifi = "wifi_ssid" in content and "my_wifi_ssid" not in content
    has_key = "api_encryption_key" in content
    return has_wifi and has_key


def _detect_wifi_ssid() -> str | None:
    """Detect the currently connected WiFi SSID, or None if unavailable."""
    system = platform.system()
    try:
        if system == "Linux":
            out = subprocess.run(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                if line.startswith("yes:"):
                    return line.split(":", 1)[1]
        elif system == "Darwin":
            out = subprocess.run(
                [
                    "/System/Library/PrivateFrameworks/Apple80211.framework"
                    "/Versions/Current/Resources/airport",
                    "-I",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                if " SSID:" in line:
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError, subprocess.TimeoutExpired:
        pass
    return None


def _prompt_secrets() -> bool:
    """Prompt for WiFi credentials, generate encryption key, write secrets.yaml."""
    import base64
    import secrets

    secrets_path = ESPHOME_DIR / "secrets.yaml"
    console.print(
        "  WiFi credentials are needed for device firmware.\n"
        f"  They will be saved to [bold]{secrets_path}[/bold]\n"
    )
    detected_ssid = _detect_wifi_ssid()
    if detected_ssid:
        ssid = click.prompt("  WiFi SSID", default=detected_ssid)
    else:
        ssid = click.prompt("  WiFi SSID")
    password = click.prompt("  WiFi password", hide_input=True)
    if not ssid:
        return False

    # Read existing secrets to preserve any extra keys
    existing_keys: dict[str, str] = {}
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if ":" in line and not line.startswith("#"):
                key = line.split(":", 1)[0].strip()
                existing_keys[key] = line

    lines = [f'wifi_ssid: "{ssid}"', f'wifi_password: "{password}"']

    # Generate encryption key if not already present
    if "api_encryption_key" in existing_keys:
        lines.append(existing_keys["api_encryption_key"])
    else:
        key = base64.b64encode(secrets.token_bytes(32)).decode()
        lines.append(f'api_encryption_key: "{key}"')
        console.print(f"  Generated API encryption key: [dim]{key}[/dim]")

    # Preserve any other existing keys
    for k, v in existing_keys.items():
        if k not in ("wifi_ssid", "wifi_password", "api_encryption_key"):
            lines.append(v)

    secrets_path.write_text("\n".join(lines) + "\n")
    console.print(f"  [green]Saved to {secrets_path}[/green]\n")
    return True


def _detect_serial_ports() -> list[dict]:
    """Detect available serial ports, filtering out generic ttyS* ports."""
    try:
        import serial.tools.list_ports

        ports = []
        for p in serial.tools.list_ports.comports():
            # Skip generic ttyS ports (not real USB devices)
            if re.match(r"/dev/ttyS\d+", p.device):
                continue
            ports.append({"device": p.device, "description": p.description or p.device})
        return ports
    except ImportError:
        return []


def _get_device_name(config_path: Path) -> str | None:
    """Extract the ESPHome device name from a YAML config."""
    try:
        content = config_path.read_text()
        # Use a simple regex to avoid yaml parsing !secret tags
        match = re.search(r"^\s+name:\s+(\S+)", content, re.MULTILINE)
        return match.group(1) if match else None
    except Exception:
        return None


def _scan_and_add_device(device_name: str) -> None:
    """Scan for a newly flashed device and add it to the Ovi config."""
    from ovi_voice_assistant.discovery import discover_devices

    console.print("  Scanning for the device on the network...")
    devices = asyncio.run(discover_devices(timeout=10.0))

    # Find devices matching the name prefix (e.g., "voice-pe" matches "voice-pe-abcdef")
    matches = [d for d in devices if d["name"].startswith(device_name)]
    if not matches:
        console.print(
            "  [yellow]Device not found on the network yet.[/yellow]\n"
            "  It may still be booting. Run [bold]ovi --scan[/bold] in a moment."
        )
        return

    device = matches[0]
    host = device["host"]
    console.print(f"  Found: [green]{device['name']}[/green] ({device['ip']})")

    from ovi_voice_assistant.setup import add_devices_to_config

    add_devices_to_config([host])
    console.print(f"  [green]Added {host} to config[/green]")


def run_flash() -> None:
    """Run the interactive device flashing flow."""
    console.print()
    console.print(
        Panel(
            "[bold]Ovi — Device Flashing[/bold]\n"
            "Compile and flash ESPHome firmware to a device",
            expand=False,
        )
    )

    # Check esphome directory exists
    if not ESPHOME_DIR.is_dir():
        console.print(
            f"  [red]ESPHome directory not found:[/red] {ESPHOME_DIR}\n"
            "  Run this command from the Ovi project root."
        )
        return

    # Find device configs
    configs = _find_device_configs()
    if not configs:
        console.print("  [red]No device configs found in esphome/[/red]")
        return

    # Check WiFi credentials
    if not _check_secrets():
        console.print("  [yellow]WiFi credentials not configured.[/yellow]\n")
        if not _prompt_secrets():
            console.print("  [red]WiFi credentials required for flashing.[/red]")
            return

    # Select device
    console.print("\n  [bold]Select a device to flash:[/bold]\n")
    for i, cfg in enumerate(configs, 1):
        console.print(
            f"    [cyan]{i}.[/cyan] {cfg['description']} [dim]({cfg['name']}.yaml)[/dim]"
        )

    while True:
        raw = click.prompt("\n  Device number", type=str)
        try:
            idx = int(raw)
            if 1 <= idx <= len(configs):
                selected = configs[idx - 1]
                break
        except ValueError:
            pass
        console.print(f"  [red]Enter a number 1-{len(configs)}[/red]")

    console.print(f"\n  Selected: [bold]{selected['description']}[/bold]")

    # Select flash method
    console.print("\n  [bold]Flash method:[/bold]\n")
    console.print(
        "    [cyan]1.[/cyan] USB — flash over serial (first time or recovery)"
    )
    console.print("    [cyan]2.[/cyan] OTA — flash over WiFi (device already running)")

    method = click.prompt("\n  Method", type=click.Choice(["1", "2"]), default="1")

    # Build esphome command
    cmd = [sys.executable, "-m", "esphome", "run", str(selected["path"]), "--no-logs"]

    if method == "2":
        # OTA — let esphome resolve via mDNS
        cmd.extend(["--device", "OTA"])
        console.print("\n  Using OTA — device must be on the network.")
    else:
        # USB — detect serial ports
        ports = _detect_serial_ports()
        if ports:
            console.print("\n  [bold]Serial ports detected:[/bold]\n")
            for i, port in enumerate(ports, 1):
                console.print(
                    f"    [cyan]{i}.[/cyan] {port['device']} — {port['description']}"
                )

            while True:
                raw = click.prompt(
                    "\n  Port number (or Enter for auto-detect)",
                    default="",
                    show_default=False,
                )
                if not raw:
                    break
                try:
                    idx = int(raw)
                    if 1 <= idx <= len(ports):
                        cmd.extend(["--device", ports[idx - 1]["device"]])
                        break
                except ValueError:
                    pass
                console.print(f"  [red]Enter a number 1-{len(ports)}[/red]")
        else:
            console.print("\n  [yellow]No serial ports detected.[/yellow]")
            console.print("  Make sure the device is plugged in via USB.")
            if not click.confirm("  Continue anyway?", default=True):
                return

    # Run esphome
    console.print()
    console.rule("[bold]Compiling and flashing[/bold]")
    console.print(f"  Running: [dim]{' '.join(cmd)}[/dim]\n")

    result = subprocess.run(cmd)

    console.print()
    if result.returncode == 0:
        console.print("  [green]Flash complete![/green]")
        console.print("  The device will reboot and connect to WiFi.\n")
        device_name = _get_device_name(selected["path"])
        if device_name:
            _scan_and_add_device(device_name)
    else:
        console.print(f"  [red]Flash failed (exit code {result.returncode})[/red]")
        if method == "1":
            console.print(
                "  Tips:\n"
                "  - Hold the BOOT button while plugging in USB\n"
                "  - Check that the serial port is not in use\n"
                "  - Try a different USB cable (data, not charge-only)"
            )
