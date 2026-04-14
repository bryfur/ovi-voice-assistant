"""Tests for the retain module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ovi_voice_assistant.memory.retain import _build_fact_text, retain
from ovi_voice_assistant.memory.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    s.open()
    yield s
    s.close()


@pytest.fixture
def embedder():
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return mock


@pytest.fixture
def llm():
    mock = AsyncMock()
    choice = MagicMock()
    choice.message.content = """[
        {
            "what": "Alice is allergic to shellfish",
            "who": "Alice",
            "where": "",
            "when": "",
            "why": "dietary restriction",
            "fact_type": "experience",
            "confidence": 0.95,
            "entities": [{"name": "Alice", "type": "person"}]
        }
    ]"""
    response = MagicMock()
    response.choices = [choice]
    mock.chat.completions.create = AsyncMock(return_value=response)
    return mock


def test_build_fact_text():
    raw = {"what": "Alice likes pizza", "who": "Alice", "where": "NYC", "when": "", "why": ""}
    text = _build_fact_text(raw)
    assert "Alice likes pizza" in text
    assert "who: Alice" in text
    assert "where: NYC" in text


def test_build_fact_text_minimal():
    text = _build_fact_text({"what": "Simple fact"})
    assert text == "Simple fact"


@pytest.mark.asyncio
async def test_retain_extracts_facts(store, embedder, llm):
    result = await retain(
        bank_id="test",
        content="Alice told me she's allergic to shellfish",
        llm=llm,
        llm_model="test-model",
        embedder=embedder,
        store=store,
    )
    assert result.success
    assert result.facts_count == 1
    assert len(result.fact_ids) == 1

    # Verify fact was stored
    facts = store.get_facts("test")
    assert len(facts) == 1
    assert "allergic to shellfish" in facts[0].what


@pytest.mark.asyncio
async def test_retain_extracts_entities(store, embedder, llm):
    result = await retain(
        bank_id="test",
        content="Alice told me she's allergic to shellfish",
        llm=llm,
        llm_model="test-model",
        embedder=embedder,
        store=store,
    )
    assert len(result.entity_ids) == 1

    entities = store.get_entities("test")
    assert len(entities) == 1
    assert entities[0].text == "Alice"


@pytest.mark.asyncio
async def test_retain_empty_extraction(store, embedder, llm):
    choice = MagicMock()
    choice.message.content = "[]"
    response = MagicMock()
    response.choices = [choice]
    llm.chat.completions.create = AsyncMock(return_value=response)

    result = await retain(
        bank_id="test",
        content="Hello, how are you?",
        llm=llm,
        llm_model="test-model",
        embedder=embedder,
        store=store,
    )
    assert result.success
    assert result.facts_count == 0


@pytest.mark.asyncio
async def test_retain_handles_llm_failure(store, embedder, llm):
    llm.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    result = await retain(
        bank_id="test",
        content="some content",
        llm=llm,
        llm_model="test-model",
        embedder=embedder,
        store=store,
    )
    assert not result.success


@pytest.mark.asyncio
async def test_retain_handles_invalid_json(store, embedder, llm):
    choice = MagicMock()
    choice.message.content = "not valid json"
    response = MagicMock()
    response.choices = [choice]
    llm.chat.completions.create = AsyncMock(return_value=response)

    result = await retain(
        bank_id="test",
        content="some content",
        llm=llm,
        llm_model="test-model",
        embedder=embedder,
        store=store,
    )
    assert not result.success


@pytest.mark.asyncio
async def test_retain_entity_dedup(store, embedder, llm):
    """Retaining twice with same entity should merge fact_ids."""
    # First retain — creates "Alice" entity
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    await retain(
        bank_id="test", content="first", llm=llm, llm_model="m",
        embedder=embedder, store=store,
    )

    # Second retain — "Alice" entity should merge
    embedder.embed = AsyncMock(return_value=[[0.4, 0.5, 0.6]])
    await retain(
        bank_id="test", content="second", llm=llm, llm_model="m",
        embedder=embedder, store=store,
    )

    entities = store.get_entities("test")
    assert len(entities) == 1  # Only one "Alice"
    assert len(entities[0].fact_ids) == 2  # Both facts linked
