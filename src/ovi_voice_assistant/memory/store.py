"""SQLite memory store with numpy vector search."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from ovi_voice_assistant.memory.types import Entity, EntityType, Fact, FactType

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL,
    what TEXT NOT NULL,
    who TEXT DEFAULT '',
    "where" TEXT DEFAULT '',
    "when" TEXT DEFAULT '',
    why TEXT DEFAULT '',
    fact_type TEXT DEFAULT 'world',
    confidence REAL DEFAULT 1.0,
    embedding TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    occurred_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_facts_bank ON facts(bank_id);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL,
    entity_type TEXT DEFAULT 'other',
    embedding TEXT DEFAULT '[]',
    fact_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_bank ON entities(bank_id);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(bank_id, text COLLATE NOCASE);
"""


def _fact_from_row(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        bank_id=row["bank_id"],
        text=row["text"],
        what=row["what"],
        who=row["who"],
        where=row["where"],
        when=row["when"],
        why=row["why"],
        fact_type=FactType(row["fact_type"]),
        confidence=row["confidence"],
        embedding=json.loads(row["embedding"]),
        created_at=row["created_at"],
        occurred_at=row["occurred_at"],
    )


def _entity_from_row(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        bank_id=row["bank_id"],
        text=row["text"],
        entity_type=EntityType(row["entity_type"]),
        embedding=json.loads(row["embedding"]),
        fact_ids=json.loads(row["fact_ids"]),
        created_at=row["created_at"],
    )


class MemoryStore:
    """SQLite-backed memory store with in-process vector search."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        path = Path(self._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        logger.info("Memory store opened: %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store not opened")
        return self._conn

    # -- Facts --

    def save_facts(self, facts: list[Fact]) -> None:
        self._db.executemany(
            """INSERT OR REPLACE INTO facts
               (id, bank_id, text, what, who, "where", "when", why,
                fact_type, confidence, embedding, created_at, occurred_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    f.id, f.bank_id, f.text, f.what, f.who, f.where, f.when, f.why,
                    f.fact_type.value, f.confidence, json.dumps(f.embedding),
                    f.created_at, f.occurred_at,
                )
                for f in facts
            ],
        )
        self._db.commit()

    def get_facts(self, bank_id: str, fact_ids: list[str] | None = None) -> list[Fact]:
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            rows = self._db.execute(
                f"SELECT * FROM facts WHERE bank_id=? AND id IN ({placeholders})",
                [bank_id, *fact_ids],
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE bank_id=?", (bank_id,)
            ).fetchall()
        return [_fact_from_row(r) for r in rows]

    def search_facts_by_embedding(
        self, bank_id: str, query_embedding: list[float], limit: int = 50
    ) -> list[tuple[Fact, float]]:
        """Return facts ranked by cosine similarity to query embedding."""
        rows = self._db.execute(
            "SELECT * FROM facts WHERE bank_id=?", (bank_id,)
        ).fetchall()
        if not rows or not query_embedding:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scored: list[tuple[Fact, float]] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            if not emb:
                continue
            v = np.array(emb, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((_fact_from_row(row), sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def search_facts_by_text(
        self, bank_id: str, query: str, limit: int = 50
    ) -> list[Fact]:
        """Keyword search using LIKE on fact text."""
        terms = query.lower().split()
        if not terms:
            return []
        # Match facts containing any query term
        conditions = " OR ".join(["LOWER(text) LIKE ?" for _ in terms])
        params = [bank_id, *[f"%{t}%" for t in terms]]
        rows = self._db.execute(
            f"SELECT * FROM facts WHERE bank_id=? AND ({conditions}) LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [_fact_from_row(r) for r in rows]

    # -- Entities --

    def save_entities(self, entities: list[Entity]) -> None:
        self._db.executemany(
            """INSERT OR REPLACE INTO entities
               (id, bank_id, text, entity_type, embedding, fact_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    e.id, e.bank_id, e.text, e.entity_type.value,
                    json.dumps(e.embedding), json.dumps(e.fact_ids), e.created_at,
                )
                for e in entities
            ],
        )
        self._db.commit()

    def get_entities(self, bank_id: str) -> list[Entity]:
        rows = self._db.execute(
            "SELECT * FROM entities WHERE bank_id=?", (bank_id,)
        ).fetchall()
        return [_entity_from_row(r) for r in rows]

    def get_entity_by_text(
        self, bank_id: str, text: str, entity_type: EntityType | None = None
    ) -> Entity | None:
        if entity_type:
            row = self._db.execute(
                "SELECT * FROM entities WHERE bank_id=? AND text=? COLLATE NOCASE AND entity_type=?",
                (bank_id, text, entity_type.value),
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM entities WHERE bank_id=? AND text=? COLLATE NOCASE",
                (bank_id, text),
            ).fetchone()
        return _entity_from_row(row) if row else None

    def get_facts_for_entity(self, bank_id: str, entity_text: str) -> list[Fact]:
        """Get all facts linked to an entity by name."""
        entity = self.get_entity_by_text(bank_id, entity_text)
        if not entity or not entity.fact_ids:
            return []
        return self.get_facts(bank_id, entity.fact_ids)
