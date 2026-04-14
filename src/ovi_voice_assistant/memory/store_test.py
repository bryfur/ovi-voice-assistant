"""Tests for the SQLite memory store."""

import pytest

from ovi_voice_assistant.memory.store import MemoryStore
from ovi_voice_assistant.memory.types import Entity, EntityType, Fact, FactType


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    s.open()
    yield s
    s.close()


def _make_fact(id="f1", bank_id="test", text="Alice likes pizza", embedding=None, **kw):
    return Fact(
        id=id,
        bank_id=bank_id,
        text=text,
        what=text,
        embedding=embedding or [],
        created_at="2025-01-01T00:00:00+00:00",
        **kw,
    )


def test_save_and_get_facts(store):
    f = _make_fact()
    store.save_facts([f])
    facts = store.get_facts("test")
    assert len(facts) == 1
    assert facts[0].text == "Alice likes pizza"
    assert facts[0].fact_type == FactType.WORLD


def test_get_facts_by_ids(store):
    store.save_facts([_make_fact(id="a"), _make_fact(id="b"), _make_fact(id="c")])
    facts = store.get_facts("test", ["a", "c"])
    assert {f.id for f in facts} == {"a", "c"}


def test_search_by_text(store):
    store.save_facts(
        [
            _make_fact(id="1", text="Alice likes pizza"),
            _make_fact(id="2", text="Bob likes tacos"),
            _make_fact(id="3", text="Carol likes sushi"),
        ]
    )
    results = store.search_facts_by_text("test", "alice pizza")
    ids = {f.id for f in results}
    assert "1" in ids


def test_search_by_embedding(store):
    store.save_facts(
        [
            _make_fact(id="1", text="cat", embedding=[1.0, 0.0, 0.0]),
            _make_fact(id="2", text="dog", embedding=[0.9, 0.1, 0.0]),
            _make_fact(id="3", text="car", embedding=[0.0, 0.0, 1.0]),
        ]
    )
    results = store.search_facts_by_embedding("test", [1.0, 0.0, 0.0], limit=2)
    assert len(results) == 2
    # First result should be the exact match
    assert results[0][0].id == "1"
    assert results[0][1] == pytest.approx(1.0, abs=0.01)


def test_search_by_embedding_empty_store(store):
    results = store.search_facts_by_embedding("test", [1.0, 0.0])
    assert results == []


def test_save_and_get_entities(store):
    e = Entity(
        id="e1",
        bank_id="test",
        text="Alice",
        entity_type=EntityType.PERSON,
        fact_ids=["f1"],
        created_at="2025-01-01T00:00:00+00:00",
    )
    store.save_entities([e])
    entities = store.get_entities("test")
    assert len(entities) == 1
    assert entities[0].text == "Alice"
    assert entities[0].entity_type == EntityType.PERSON


def test_get_entity_by_text_case_insensitive(store):
    e = Entity(
        id="e1",
        bank_id="test",
        text="Alice",
        entity_type=EntityType.PERSON,
        fact_ids=["f1"],
        created_at="2025-01-01T00:00:00+00:00",
    )
    store.save_entities([e])
    found = store.get_entity_by_text("test", "alice")
    assert found is not None
    assert found.text == "Alice"


def test_get_facts_for_entity(store):
    store.save_facts(
        [
            _make_fact(id="f1", text="Alice likes pizza"),
            _make_fact(id="f2", text="Bob likes tacos"),
        ]
    )
    store.save_entities(
        [
            Entity(
                id="e1",
                bank_id="test",
                text="Alice",
                entity_type=EntityType.PERSON,
                fact_ids=["f1"],
                created_at="2025-01-01T00:00:00+00:00",
            )
        ]
    )
    facts = store.get_facts_for_entity("test", "Alice")
    assert len(facts) == 1
    assert facts[0].id == "f1"
