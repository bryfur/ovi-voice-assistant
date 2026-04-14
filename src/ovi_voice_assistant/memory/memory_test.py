"""Tests for the Memory high-level API."""

from unittest.mock import patch

import pytest

from ovi_voice_assistant.config import MemoryConfig, Settings
from ovi_voice_assistant.memory import Memory


@pytest.fixture
def settings(tmp_path):
    with patch("ovi_voice_assistant.config.CONFIG_PATH", tmp_path / "c.yaml"):
        return Settings(
            _env_file=None,
            devices="",
            memory=MemoryConfig(enabled=True, db_path=":memory:"),
        )


@pytest.mark.asyncio
async def test_retain_raises_before_load(settings):
    mem = Memory(settings)
    with pytest.raises(RuntimeError, match="load"):
        await mem.retain("test")


@pytest.mark.asyncio
async def test_recall_raises_before_load(settings):
    mem = Memory(settings)
    with pytest.raises(RuntimeError, match="load"):
        await mem.recall("test")


@patch("ovi_voice_assistant.memory.memory.Embedder")
def test_load_opens_store(mock_embedder_cls, settings):
    mem = Memory(settings)
    mem.load()
    assert mem.bank_id == "voice-assistant"
    mock_embedder_cls.return_value.load.assert_called_once()
    mem.close()


@patch("ovi_voice_assistant.memory.memory.Embedder")
def test_close_is_idempotent(mock_embedder_cls, settings):
    mem = Memory(settings)
    mem.load()
    mem.close()
    mem.close()  # should not raise
