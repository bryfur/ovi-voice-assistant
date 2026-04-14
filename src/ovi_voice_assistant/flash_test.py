"""Tests for the device flashing module."""

import subprocess
from unittest.mock import MagicMock, patch

import yaml

from ovi_voice_assistant.flash import (
    _check_secrets,
    _detect_serial_ports,
    _detect_wifi_ssid,
    _find_device_configs,
    _get_device_name,
    _scan_and_add_device,
)


class TestFindDeviceConfigs:
    def test_finds_yaml_files(self, tmp_path):
        (tmp_path / "voice-pe.yaml").write_text(
            "# Ovi — Open Voice Assistant — ESPHome config for Voice PE\n"
        )
        (tmp_path / "atom-echo.yaml").write_text(
            "# Ovi — Open Voice Assistant — ESPHome config for ATOM Echo\n"
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            configs = _find_device_configs()

        assert len(configs) == 2
        assert configs[0]["name"] == "voice-pe"
        assert configs[0]["description"] == "Voice PE"
        assert configs[1]["name"] == "atom-echo"
        assert configs[1]["description"] == "ATOM Echo"

    def test_excludes_secrets_yaml(self, tmp_path):
        (tmp_path / "secrets.yaml").write_text('wifi_ssid: "test"\n')
        (tmp_path / "device.yaml").write_text(
            "# Ovi — Open Voice Assistant — ESPHome config for Test Device\n"
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            configs = _find_device_configs()

        assert len(configs) == 1
        assert configs[0]["name"] == "device"

    def test_empty_directory(self, tmp_path):
        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            configs = _find_device_configs()

        assert configs == []

    def test_ble_variant_description(self, tmp_path):
        (tmp_path / "voice-pe-ble.yaml").write_text(
            "# Ovi — Open Voice Assistant — BLE firmware for Voice PE\n"
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            configs = _find_device_configs()

        assert len(configs) == 1
        assert "Voice PE" in configs[0]["description"]


class TestDetectWifiSsid:
    def test_linux_nmcli_active(self):
        result = MagicMock()
        result.stdout = "no:OtherNetwork\nyes:MyHomeWifi\n"

        with (
            patch("ovi_voice_assistant.flash.platform.system", return_value="Linux"),
            patch("ovi_voice_assistant.flash.subprocess.run", return_value=result),
        ):
            assert _detect_wifi_ssid() == "MyHomeWifi"

    def test_linux_no_active_wifi(self):
        result = MagicMock()
        result.stdout = "no:SomeNetwork\n"

        with (
            patch("ovi_voice_assistant.flash.platform.system", return_value="Linux"),
            patch("ovi_voice_assistant.flash.subprocess.run", return_value=result),
        ):
            assert _detect_wifi_ssid() is None

    def test_macos_airport(self):
        result = MagicMock()
        result.stdout = "     agrCtlRSSI: -55\n     SSID: CafeWifi\n     BSSID: aa:bb\n"

        with (
            patch("ovi_voice_assistant.flash.platform.system", return_value="Darwin"),
            patch("ovi_voice_assistant.flash.subprocess.run", return_value=result),
        ):
            assert _detect_wifi_ssid() == "CafeWifi"

    def test_nmcli_not_found(self):
        with (
            patch("ovi_voice_assistant.flash.platform.system", return_value="Linux"),
            patch(
                "ovi_voice_assistant.flash.subprocess.run",
                side_effect=FileNotFoundError,
            ),
        ):
            assert _detect_wifi_ssid() is None

    def test_timeout(self):
        with (
            patch("ovi_voice_assistant.flash.platform.system", return_value="Linux"),
            patch(
                "ovi_voice_assistant.flash.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nmcli", 5),
            ),
        ):
            assert _detect_wifi_ssid() is None

    def test_unknown_platform(self):
        with patch("ovi_voice_assistant.flash.platform.system", return_value="Windows"):
            assert _detect_wifi_ssid() is None


class TestCheckSecrets:
    def test_returns_false_when_no_file(self, tmp_path):
        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            assert _check_secrets() is False

    def test_returns_false_with_placeholder(self, tmp_path):
        (tmp_path / "secrets.yaml").write_text(
            'wifi_ssid: "my_wifi_ssid"\nwifi_password: "my_wifi_password"\n'
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            assert _check_secrets() is False

    def test_returns_false_without_encryption_key(self, tmp_path):
        (tmp_path / "secrets.yaml").write_text(
            'wifi_ssid: "MyNetwork"\nwifi_password: "secret123"\n'
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            assert _check_secrets() is False

    def test_returns_true_with_all_secrets(self, tmp_path):
        (tmp_path / "secrets.yaml").write_text(
            'wifi_ssid: "MyNetwork"\n'
            'wifi_password: "secret123"\n'
            'api_encryption_key: "abc123"\n'
        )

        with patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path):
            assert _check_secrets() is True


class TestPromptSecrets:
    def test_writes_wifi_and_generates_key(self, tmp_path):
        secrets_path = tmp_path / "secrets.yaml"

        with (
            patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path),
            patch("ovi_voice_assistant.flash.click") as mock_click,
            patch("ovi_voice_assistant.flash.console"),
        ):
            mock_click.prompt.side_effect = ["MyNetwork", "password123"]

            from ovi_voice_assistant.flash import _prompt_secrets

            result = _prompt_secrets()

        assert result is True
        content = secrets_path.read_text()
        assert 'wifi_ssid: "MyNetwork"' in content
        assert 'wifi_password: "password123"' in content
        assert "api_encryption_key:" in content

    def test_preserves_existing_encryption_key(self, tmp_path):
        secrets_path = tmp_path / "secrets.yaml"
        secrets_path.write_text('api_encryption_key: "existing-key-123"\n')

        with (
            patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path),
            patch("ovi_voice_assistant.flash.click") as mock_click,
            patch("ovi_voice_assistant.flash.console"),
        ):
            mock_click.prompt.side_effect = ["NewSSID", "newpass"]

            from ovi_voice_assistant.flash import _prompt_secrets

            result = _prompt_secrets()

        assert result is True
        content = secrets_path.read_text()
        assert 'wifi_ssid: "NewSSID"' in content
        assert "existing-key-123" in content

    def test_returns_false_on_empty_ssid(self, tmp_path):
        with (
            patch("ovi_voice_assistant.flash.ESPHOME_DIR", tmp_path),
            patch("ovi_voice_assistant.flash.click") as mock_click,
            patch("ovi_voice_assistant.flash.console"),
        ):
            mock_click.prompt.side_effect = ["", ""]

            from ovi_voice_assistant.flash import _prompt_secrets

            result = _prompt_secrets()

        assert result is False


class TestDetectSerialPorts:
    def test_filters_ttyS_ports(self):
        mock_port_usb = MagicMock()
        mock_port_usb.device = "/dev/ttyACM0"
        mock_port_usb.description = "USB JTAG/serial debug unit"

        mock_port_ttyS = MagicMock()
        mock_port_ttyS.device = "/dev/ttyS0"
        mock_port_ttyS.description = "n/a"

        with patch(
            "serial.tools.list_ports.comports",
            return_value=[mock_port_usb, mock_port_ttyS],
        ):
            ports = _detect_serial_ports()

        assert len(ports) == 1
        assert ports[0]["device"] == "/dev/ttyACM0"

    def test_returns_empty_when_no_ports(self):
        with patch("serial.tools.list_ports.comports", return_value=[]):
            ports = _detect_serial_ports()

        assert ports == []

    def test_returns_empty_when_import_fails(self):
        with patch(
            "serial.tools.list_ports.comports",
            side_effect=ImportError("no module"),
        ):
            # Import error is caught inside the function
            ports = _detect_serial_ports()

        assert ports == []


class TestGetDeviceName:
    def test_extracts_name(self, tmp_path):
        config = tmp_path / "voice-pe.yaml"
        config.write_text("esphome:\n  name: voice-pe\n  friendly_name: Voice\n")

        assert _get_device_name(config) == "voice-pe"

    def test_returns_none_for_missing_name(self, tmp_path):
        config = tmp_path / "bad.yaml"
        config.write_text("esphome:\n  friendly_name: Voice\n")

        assert _get_device_name(config) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _get_device_name(tmp_path / "nonexistent.yaml") is None


class TestScanAndAddDevice:
    def test_adds_device_preserving_existing_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "llm": {"model": "gpt-4o", "api_key": "sk-test"},
                    "transport": {"codec": "lc3"},
                }
            )
        )

        found = [
            {
                "name": "voice-pe-abcdef",
                "host": "voice-pe-abcdef.local",
                "ip": "10.0.0.5",
            }
        ]

        with (
            patch("ovi_voice_assistant.flash.asyncio.run", return_value=found),
            patch(
                "ovi_voice_assistant.setup._read_encryption_key",
                return_value="testkey123",
            ),
            patch("ovi_voice_assistant.config.CONFIG_PATH", config_path),
            patch("ovi_voice_assistant.setup.CONFIG_PATH", config_path),
            patch("ovi_voice_assistant.flash.console"),
        ):
            _scan_and_add_device("voice-pe")

        loaded = yaml.safe_load(config_path.read_text())
        assert any("voice-pe-abcdef.local" in d for d in loaded["devices"])
        assert any("testkey123" in d for d in loaded["devices"])
        assert loaded["llm"]["model"] == "gpt-4o"
        assert loaded["llm"]["api_key"] == "sk-test"
        assert loaded["transport"]["codec"] == "lc3"

    def test_skips_duplicate_device(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"devices": ["voice-pe-abcdef.local"]}))

        found = [
            {
                "name": "voice-pe-abcdef",
                "host": "voice-pe-abcdef.local",
                "ip": "10.0.0.5",
            }
        ]

        with (
            patch("ovi_voice_assistant.flash.asyncio.run", return_value=found),
            patch("ovi_voice_assistant.setup._read_encryption_key", return_value=None),
            patch("ovi_voice_assistant.config.CONFIG_PATH", config_path),
            patch("ovi_voice_assistant.setup.CONFIG_PATH", config_path),
            patch("ovi_voice_assistant.flash.console"),
        ):
            _scan_and_add_device("voice-pe")

        loaded = yaml.safe_load(config_path.read_text())
        assert len(loaded["devices"]) == 1

    def test_handles_no_devices_found(self):
        with (
            patch("ovi_voice_assistant.flash.asyncio.run", return_value=[]),
            patch("ovi_voice_assistant.flash.console") as mock_console,
        ):
            _scan_and_add_device("voice-pe")

        mock_console.print.assert_any_call(
            "  [yellow]Device not found on the network yet.[/yellow]\n"
            "  It may still be booting. Run [bold]ovi --scan[/bold] in a moment."
        )
