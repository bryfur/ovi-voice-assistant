"""Recall — multi-strategy retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum

from ovi_voice_assistant.memory.embedder import Embedder
from ovi_voice_assistant.memory.store import MemoryStore
from ovi_voice_assistant.memory.types import Entity, Fact, RecallResult

logger = logging.getLogger(__name__)

# RRF constant — standard value from the literature.
_RRF_K = 60

# Temporal decay: e^(-days/HALF_LIFE). At 30 days, score ~0.37.
_TEMPORAL_HALF_LIFE = 30.0


class Budget(StrEnum):
    """Controls how many candidates to retrieve."""

    LOW = "low"  # 50 candidates — fast
    MID = "mid"  # 150 candidates — balanced
    HIGH = "high"  # 400 candidates — thorough


_BUDGET_LIMITS = {
    Budget.LOW: 50,
    Budget.MID: 150,
    Budget.HIGH: 400,
}


async def recall(
    *,
    bank_id: str,
    query: str,
    embedder: Embedder,
    store: MemoryStore,
    budget: Budget = Budget.MID,
    max_tokens: int = 2048,
) -> RecallResult:
    """Search memory using 4 parallel strategies + RRF fusion.

    Strategies:
        1. Semantic — cosine similarity on embeddings
        2. Keyword (BM25-lite) — LIKE query on fact text
        3. Entity graph — facts linked to entities mentioned in query
        4. Temporal — exponential decay favoring recent facts

    Results are merged via Reciprocal Rank Fusion and trimmed to token budget.
    """
    limit = _BUDGET_LIMITS.get(budget, 150)

    # 1. Semantic search
    try:
        query_emb = await embedder.embed_one(query)
    except Exception:
        logger.exception("Query embedding failed")
        query_emb = []

    semantic_results: list[tuple[str, float]] = []
    if query_emb:
        scored = store.search_facts_by_embedding(bank_id, query_emb, limit=limit)
        semantic_results = [(f.id, sim) for f, sim in scored]

    # 2. Keyword search
    keyword_facts = store.search_facts_by_text(bank_id, query, limit=limit)
    keyword_results = [(f.id, 1.0) for f in keyword_facts]

    # 3. Entity graph search
    entity_results = _entity_graph_search(bank_id, query, store, limit)

    # 4. Temporal search — score all facts by recency
    temporal_results = _temporal_search(bank_id, store, limit)

    # Merge via RRF
    rrf_scores = _rrf_merge(semantic_results, keyword_results, entity_results, temporal_results)

    if not rrf_scores:
        return RecallResult(results=[], total_candidates=0)

    # Rank by RRF score
    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # Fetch full facts and apply token budget
    all_facts = store.get_facts(bank_id, ranked_ids[:limit])
    fact_map = {f.id: f for f in all_facts}
    ranked_facts = [fact_map[fid] for fid in ranked_ids if fid in fact_map]

    # Set relevance scores
    for fact in ranked_facts:
        fact.relevance = rrf_scores.get(fact.id, 0.0)

    # Apply token budget
    budgeted = _apply_token_budget(ranked_facts, max_tokens)

    # Gather relevant entities
    entities = _gather_entities(bank_id, query, store)

    return RecallResult(
        results=budgeted,
        entities=entities,
        total_candidates=len(rrf_scores),
    )


def _entity_graph_search(
    bank_id: str, query: str, store: MemoryStore, limit: int
) -> list[tuple[str, float]]:
    """Find facts linked to entities mentioned in the query."""
    entities = store.get_entities(bank_id)
    if not entities:
        return []

    query_lower = query.lower()
    matching_entities = [e for e in entities if e.text.lower() in query_lower]

    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for entity in matching_entities:
        for fid in entity.fact_ids:
            if fid not in seen:
                seen.add(fid)
                results.append((fid, 1.0))
                if len(results) >= limit:
                    return results
    return results


def _temporal_search(
    bank_id: str, store: MemoryStore, limit: int
) -> list[tuple[str, float]]:
    """Score all facts by recency using exponential decay."""
    facts = store.get_facts(bank_id)
    if not facts:
        return []

    now = datetime.now(UTC)
    scored: list[tuple[str, float]] = []

    for fact in facts:
        if not fact.created_at:
            continue
        try:
            created = datetime.fromisoformat(fact.created_at)
            delta_days = (now - created).total_seconds() / 86400.0
            score = math.exp(-delta_days / _TEMPORAL_HALF_LIFE)
            scored.append((fact.id, score))
        except (ValueError, TypeError):
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _rrf_merge(*ranked_lists: list[tuple[str, float]]) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d)))."""
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (fact_id, _) in enumerate(ranked_list):
            scores[fact_id] += 1.0 / (_RRF_K + rank + 1)
    return dict(scores)


def _apply_token_budget(facts: list[Fact], max_tokens: int) -> list[Fact]:
    """Keep facts until the token budget is exceeded (~4 chars/token)."""
    result: list[Fact] = []
    token_count = 0
    for fact in facts:
        fact_tokens = len(fact.text) // 4
        if token_count + fact_tokens > max_tokens:
            break
        result.append(fact)
        token_count += fact_tokens
    return result


def _gather_entities(bank_id: str, query: str, store: MemoryStore) -> list[Entity]:
    """Return entities mentioned in the query."""
    entities = store.get_entities(bank_id)
    query_lower = query.lower()
    return [e for e in entities if e.text.lower() in query_lower]
