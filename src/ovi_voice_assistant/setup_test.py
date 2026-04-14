"""Tests for the setup wizard."""

from unittest.mock import patch

import yaml

from ovi_voice_assistant.setup import needs_setup, save_config


class TestNeedsSetup:
    def test_returns_true_when_no_config(self, tmp_path):
        with patch("ovi_voice_assistant.setup.CONFIG_PATH", tmp_path / "config.yaml"):
            assert needs_setup() is True

    def test_returns_false_when_config_exists(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  model: test")

        with patch("ovi_voice_assistant.setup.CONFIG_PATH", config_file):
            assert needs_setup() is False


class TestSaveConfig:
    def test_creates_directory_and_file(self, tmp_path):
        config_path = tmp_path / "subdir" / "config.yaml"
        config = {"llm": {"model": "llama3"}, "transport": {"codec": "opus"}}

        save_config(config, path=config_path)

        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded["llm"]["model"] == "llama3"
        assert loaded["transport"]["codec"] == "opus"

    def test_overwrites_existing_file(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"old_key": "old_value"}))

        save_config({"llm": {"model": "new"}}, path=config_path)

        loaded = yaml.safe_load(config_path.read_text())
        assert loaded["llm"]["model"] == "new"

    def test_writes_nested_yaml(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config = {
            "llm": {"api_key": "sk-test", "model": "gpt-4o"},
            "stt": {"provider": "whisper"},
            "transport": {"codec": "lc3"},
        }

        save_config(config, path=config_path)

        text = config_path.read_text()
        loaded = yaml.safe_load(text)
        assert loaded["llm"]["model"] == "gpt-4o"
        assert loaded["transport"]["codec"] == "lc3"


def _mock_setup(config_path, prompts, confirms, existing=None):
    """Run run_setup with mocked click prompts/confirms and optional existing config."""
    with (
        patch("ovi_voice_assistant.setup.CONFIG_PATH", config_path),
        patch(
            "ovi_voice_assistant.setup._load_existing",
            return_value=existing or {},
        ),
        patch("ovi_voice_assistant.setup.click") as mock_click,
        patch("ovi_voice_assistant.setup.console"),
    ):
        prompt_iter = iter(prompts)
        confirm_iter = iter(confirms)
        mock_click.prompt.side_effect = lambda *a, **kw: next(prompt_iter)
        mock_click.confirm.side_effect = lambda *a, **kw: next(confirm_iter)
        mock_click.Choice = __import__("click").Choice

        from ovi_voice_assistant.setup import run_setup

        return run_setup()


class TestRunSetupFresh:
    def test_fresh_setup(self, tmp_path):
        config_path = tmp_path / "config.yaml"

        prompts = [
            "sk-test",  # API key
            "",  # Base URL
            "gpt-4o-mini",  # Agent model
            "whisper",  # STT provider
            "2",  # _pick: Whisper model (base.en)
            "cpu",  # STT device
            "kokoro",  # TTS provider
            "",  # _pick: Kokoro voice (af_heart default)
            "10.0.0.1",  # Device address
            "",  # _pick: Codec (lc3 default)
        ]
        confirms = [
            False,  # Don't scan
            True,  # Save
        ]

        config = _mock_setup(config_path, prompts, confirms)

        assert config["llm"]["api_key"] == "sk-test"
        assert config["stt"]["provider"] == "whisper"
        assert config["stt"]["model"] == "base.en"
        assert config["tts"]["model"] == "af_heart"
        assert config["devices"] == ["10.0.0.1"]
        assert config["transport"]["codec"] == "lc3"
        assert config_path.exists()


class TestRunSetupEdit:
    def test_keeps_existing_values_on_enter(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        existing = {
            "llm": {"api_key": "sk-existing", "model": "llama3"},
            "stt": {"provider": "whisper", "model": "small.en", "device": "cuda"},
            "tts": {"provider": "kokoro", "model": "am_adam"},
            "devices": ["voice-pe.local"],
            "transport": {"codec": "opus"},
        }

        prompts = [
            "sk-existing",  # API key
            "",  # Base URL
            "llama3",  # Agent model
            "whisper",  # STT provider
            "",  # _pick: Whisper model (small.en from existing)
            "cuda",  # STT device
            "kokoro",  # TTS provider
            "",  # _pick: Kokoro voice (am_adam from existing)
            "voice-pe.local",  # Devices
            "",  # _pick: Codec (opus from existing)
        ]
        confirms = [
            False,  # Don't scan
            True,  # Save
        ]

        config = _mock_setup(config_path, prompts, confirms, existing)

        assert config["llm"]["model"] == "llama3"
        assert config["stt"]["model"] == "small.en"
        assert config["tts"]["model"] == "am_adam"
        assert config["transport"]["codec"] == "opus"
