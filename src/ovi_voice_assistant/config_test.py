"""Tests for the config module."""

import pytest

from ovi_voice_assistant.config import DeviceConfig, Settings, parse_devices


class TestDeviceConfigDefaults:
    def test_defaults(self):
        cfg = DeviceConfig(host="10.0.0.1")

        assert cfg.host == "10.0.0.1"
        assert cfg.port == 6055
        assert cfg.encryption_key is None

    def test_custom_values(self):
        cfg = DeviceConfig(
            host="10.0.0.1",
            port=1234,
            encryption_key="abc",
        )

        assert cfg.port == 1234
        assert cfg.encryption_key == "abc"


class TestParseDevicesHostOnly:
    def test_host_only(self):
        result = parse_devices("10.0.0.1")

        assert len(result) == 1
        assert result[0].host == "10.0.0.1"
        assert result[0].port == 6055
        assert result[0].encryption_key is None


class TestParseDevicesHostPort:
    def test_host_port(self):
        result = parse_devices("10.0.0.1:1234")

        assert len(result) == 1
        assert result[0].host == "10.0.0.1"
        assert result[0].port == 1234

    def test_empty_port_uses_default(self):
        result = parse_devices("10.0.0.1:")

        assert result[0].port == 6055


class TestParseDevicesHostPortKey:
    def test_host_port_key(self):
        result = parse_devices("10.0.0.1:6055:mysecretkey")

        assert len(result) == 1
        assert result[0].encryption_key == "mysecretkey"

    def test_empty_key_is_none(self):
        result = parse_devices("10.0.0.1:6055:")

        assert result[0].encryption_key is None


class TestParseDevicesEmptyFields:
    def test_empty_port_and_key(self):
        result = parse_devices("10.0.0.1::")

        assert result[0].host == "10.0.0.1"
        assert result[0].port == 6055
        assert result[0].encryption_key is None


class TestParseDevicesMultiple:
    def test_two_devices(self):
        result = parse_devices("10.0.0.1,10.0.0.2")

        assert len(result) == 2
        assert result[0].host == "10.0.0.1"
        assert result[1].host == "10.0.0.2"

    def test_multiple_with_full_config(self):
        raw = "10.0.0.1:6055:key1,10.0.0.2:1234:key2"

        result = parse_devices(raw)

        assert len(result) == 2
        assert result[0].host == "10.0.0.1"
        assert result[0].encryption_key == "key1"
        assert result[1].host == "10.0.0.2"
        assert result[1].port == 1234
        assert result[1].encryption_key == "key2"

    def test_trailing_comma_ignored(self):
        result = parse_devices("10.0.0.1,")

        assert len(result) == 1


class TestParseDevicesEmptyAndWhitespace:
    def test_empty_string(self):
        result = parse_devices("")

        assert result == []

    def test_only_whitespace(self):
        result = parse_devices("   ")

        assert result == []

    def test_only_commas(self):
        result = parse_devices(",,,")

        assert result == []

    def test_whitespace_around_entries(self):
        result = parse_devices("  10.0.0.1 , 10.0.0.2  ")

        assert len(result) == 2
        assert result[0].host == "10.0.0.1"
        assert result[1].host == "10.0.0.2"


class TestParseDevicesErrors:
    def test_non_integer_port(self):
        with pytest.raises(ValueError):
            parse_devices("10.0.0.1:notaport")


class TestSettingsDefaults:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("OVI_DEVICES", raising=False)
        monkeypatch.delenv("OVI_TRANSPORT", raising=False)
        monkeypatch.delenv("OVI_CODEC", raising=False)
        monkeypatch.delenv("OVI_STT_PROVIDER", raising=False)
        monkeypatch.delenv("OVI_TTS_PROVIDER", raising=False)

        s = Settings(_env_file=None)

        assert s.devices == ""
        assert s.transport == "wifi"
        assert s.codec == "lc3"
        assert s.speaker_sample_rate == 0
        assert s.stt_provider == "whisper"
        assert s.tts_provider == "kokoro"
        assert s.stt_model == "base.en"
        assert s.stt_device == "cpu"
        assert s.stt_language == "en"
        assert s.stt_beam_size == 1
        assert s.stt_compute_type == "int8"
        assert s.tts_model == "af_heart"
        assert s.tts_speaker_id is None
        assert s.tts_length_scale == 1.0
        assert s.tts_sentence_silence == 0.1
        assert s.openai_api_key == ""
        assert s.openai_base_url == ""
        assert s.agent_model == "gpt-4o-mini"
        assert s.mic_sample_rate == 16000
        assert s.mic_sample_width == 2
        assert s.mic_channels == 1
        assert s.ble_device_name is None
        assert s.ble_device_address is None


class TestSettingsGetDevices:
    def test_empty_devices(self, monkeypatch):
        monkeypatch.delenv("OVI_DEVICES", raising=False)
        s = Settings(_env_file=None, devices="")

        result = s.get_devices()

        assert result == []

    def test_with_device_string(self, monkeypatch):
        monkeypatch.delenv("OVI_DEVICES", raising=False)
        s = Settings(_env_file=None, devices="10.0.0.1:6055:key")

        devices = s.get_devices()

        assert len(devices) == 1
        assert devices[0].host == "10.0.0.1"
        assert devices[0].port == 6055
        assert devices[0].encryption_key == "key"

    def test_with_multiple_devices(self, monkeypatch):
        monkeypatch.delenv("OVI_DEVICES", raising=False)
        s = Settings(_env_file=None, devices="10.0.0.1,10.0.0.2:1234")

        devices = s.get_devices()

        assert len(devices) == 2
        assert devices[1].port == 1234
