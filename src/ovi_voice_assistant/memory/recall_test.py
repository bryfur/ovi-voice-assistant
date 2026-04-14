"""Tests for the recall module."""

from unittest.mock import AsyncMock

import pytest

from ovi_voice_assistant.memory.recall import Budget, _rrf_merge, recall
from ovi_voice_assistant.memory.store import MemoryStore
from ovi_voice_assistant.memory.types import Entity, EntityType, Fact


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    s.open()
    yield s
    s.close()


@pytest.fixture
def embedder():
    mock = AsyncMock()
    mock.embed_one = AsyncMock(return_value=[1.0, 0.0, 0.0])
    return mock


def _make_fact(id, text, embedding=None):
    return Fact(
        id=id, bank_id="test", text=text, what=text,
        embedding=embedding or [1.0, 0.0, 0.0],
        created_at="2025-01-01T00:00:00+00:00",
    )


def test_rrf_merge_basic():
    list1 = [("a", 0.9), ("b", 0.8)]
    list2 = [("b", 1.0), ("c", 0.5)]
    scores = _rrf_merge(list1, list2)
    # "b" appears in both lists → higher score
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rrf_merge_empty():
    scores = _rrf_merge([], [], [])
    assert scores == {}


@pytest.mark.asyncio
async def test_recall_returns_ranked_facts(store, embedder):
    store.save_facts([
        _make_fact("1", "Alice likes pizza", [1.0, 0.0, 0.0]),
        _make_fact("2", "Bob likes tacos", [0.0, 1.0, 0.0]),
    ])
    result = await recall(
        bank_id="test", query="alice pizza", embedder=embedder,
        store=store, budget=Budget.LOW,
    )
    assert len(result.results) > 0
    # The fact matching both semantic and keyword should rank higher
    assert result.results[0].id == "1"


@pytest.mark.asyncio
async def test_recall_empty_store(store, embedder):
    result = await recall(
        bank_id="test", query="anything", embedder=embedder,
        store=store, budget=Budget.LOW,
    )
    assert result.results == []
    assert result.total_candidates == 0


@pytest.mark.asyncio
async def test_recall_entity_graph(store, embedder):
    store.save_facts([
        _make_fact("1", "Alice works at Google", [0.5, 0.5, 0.0]),
        _make_fact("2", "Bob works at Meta", [0.0, 0.5, 0.5]),
    ])
    store.save_entities([Entity(
        id="e1", bank_id="test", text="Alice",
        entity_type=EntityType.PERSON, fact_ids=["1"],
        created_at="2025-01-01T00:00:00+00:00",
    )])
    result = await recall(
        bank_id="test", query="Tell me about Alice",
        embedder=embedder, store=store, budget=Budget.LOW,
    )
    # Entity graph search should boost Alice's fact
    assert any(f.id == "1" for f in result.results)
    assert len(result.entities) == 1
    assert result.entities[0].text == "Alice"


@pytest.mark.asyncio
async def test_recall_token_budget(store, embedder):
    # Create facts with long text
    store.save_facts([
        _make_fact(str(i), f"fact number {i} " * 50, [1.0, 0.0, 0.0])
        for i in range(20)
    ])
    result = await recall(
        bank_id="test", query="fact", embedder=embedder,
        store=store, budget=Budget.HIGH, max_tokens=100,
    )
    # Should be limited by token budget, not return all 20
    assert len(result.results) < 20
    assert result.total_candidates >= len(result.results)
