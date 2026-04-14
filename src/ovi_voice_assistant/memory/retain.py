"""Retain — extract facts from text and store them in memory."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from openai import AsyncOpenAI

from ovi_voice_assistant.memory.embedder import Embedder
from ovi_voice_assistant.memory.store import MemoryStore
from ovi_voice_assistant.memory.types import (
    Entity,
    EntityType,
    Fact,
    FactType,
    RetainResult,
)

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """\
You extract structured facts from conversation text.

For each distinct fact, output a JSON object with these fields:
- "what": The core fact in 1-2 sentences. Resolve pronouns to names.
- "who": People involved (empty string if none).
- "where": Location (empty string if none).
- "when": Temporal info (empty string if none). Convert relative dates \
using today's date: {today}.
- "why": Context or significance (empty string if none).
- "fact_type": One of "world", "experience", or "assistant".
  - "world": objective facts about the world
  - "experience": user's experiences, preferences, or personal info
  - "assistant": things the assistant said or did
- "confidence": 0.0 to 1.0, how confident this fact is.
- "entities": List of {{"name": "...", "type": "person|organization|location|concept|other"}}.

Only extract facts worth remembering long-term. Skip pleasantries, \
filler, and transient statements.

Output a JSON array of fact objects. If nothing is worth extracting, output [].
"""

_EXTRACT_USER = "Extract facts from this conversation:\n\n{content}"


def _build_fact_text(raw: dict) -> str:
    """Combine fact dimensions into a single text for embedding."""
    parts = [raw.get("what", "")]
    for key in ("who", "where", "when", "why"):
        val = raw.get(key, "").strip()
        if val:
            parts.append(f"{key}: {val}")
    return ". ".join(parts)


def _parse_entity_type(raw: str) -> EntityType:
    try:
        return EntityType(raw.lower())
    except ValueError:
        return EntityType.OTHER


async def retain(
    *,
    bank_id: str,
    content: str,
    llm: AsyncOpenAI,
    llm_model: str,
    embedder: Embedder,
    store: MemoryStore,
    context: str = "",
) -> RetainResult:
    """Extract facts from content and store them in memory.

    Args:
        bank_id: Memory partition (e.g. "voice-assistant").
        content: Text to extract facts from.
        llm: OpenAI-compatible client for fact extraction.
        llm_model: Model name for the LLM.
        embedder: Embedding client.
        store: Memory store.
        context: Optional context about where this content came from.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    system_prompt = _EXTRACT_SYSTEM.format(today=today)
    user_prompt = _EXTRACT_USER.format(content=content)
    if context:
        user_prompt += f"\n\nContext: {context}"

    # Extract facts via LLM
    try:
        response = await llm.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content or "[]"
    except Exception:
        logger.exception("LLM fact extraction failed")
        return RetainResult(success=False, facts_count=0)

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        raw_facts = json.loads(cleaned)
        if not isinstance(raw_facts, list):
            raw_facts = [raw_facts]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM response as JSON: %s", raw_text[:200])
        return RetainResult(success=False, facts_count=0)

    if not raw_facts:
        return RetainResult(success=True, facts_count=0)

    now = datetime.now(UTC).isoformat()

    # Build Fact objects
    facts: list[Fact] = []
    all_raw_entities: list[tuple[str, dict]] = []  # (fact_id, entity_dict)

    for raw in raw_facts:
        fact_id = uuid.uuid4().hex[:12]
        text = _build_fact_text(raw)
        facts.append(
            Fact(
                id=fact_id,
                bank_id=bank_id,
                text=text,
                what=raw.get("what", ""),
                who=raw.get("who", ""),
                where=raw.get("where", ""),
                when=raw.get("when", ""),
                why=raw.get("why", ""),
                fact_type=FactType(raw.get("fact_type", "world")),
                confidence=float(raw.get("confidence", 1.0)),
                created_at=now,
                occurred_at=raw.get("when", ""),
            )
        )
        for ent in raw.get("entities", []):
            all_raw_entities.append((fact_id, ent))

    # Embed all fact texts in one batch
    try:
        embeddings = await embedder.embed([f.text for f in facts])
        for fact, emb in zip(facts, embeddings, strict=True):
            fact.embedding = emb
    except Exception:
        logger.exception("Embedding failed, storing facts without embeddings")

    # Store facts
    store.save_facts(facts)
    fact_ids = [f.id for f in facts]

    # Resolve and store entities
    entity_ids = await _resolve_entities(
        bank_id, all_raw_entities, embedder, store, now
    )

    logger.info("Retained %d facts, %d entities (bank=%s)", len(facts), len(entity_ids), bank_id)
    return RetainResult(
        success=True, facts_count=len(facts), fact_ids=fact_ids, entity_ids=entity_ids
    )


async def _resolve_entities(
    bank_id: str,
    raw_entities: list[tuple[str, dict]],
    embedder: Embedder,
    store: MemoryStore,
    now: str,
) -> list[str]:
    """Deduplicate and merge entities with existing ones in the store."""
    if not raw_entities:
        return []

    # Group by (name_lower, type) to deduplicate within this batch
    merged: dict[tuple[str, EntityType], tuple[str, EntityType, list[str]]] = {}
    for fact_id, ent_raw in raw_entities:
        name = ent_raw.get("name", "").strip()
        if not name:
            continue
        etype = _parse_entity_type(ent_raw.get("type", "other"))
        key = (name.lower(), etype)
        if key in merged:
            merged[key][2].append(fact_id)
        else:
            merged[key] = (name, etype, [fact_id])

    # Resolve against existing entities
    entities_to_save: list[Entity] = []
    entity_ids: list[str] = []

    for (name_lower, etype), (name, _, fact_ids) in merged.items():
        existing = store.get_entity_by_text(bank_id, name_lower, etype)
        if existing:
            # Merge fact_ids into existing entity
            new_ids = set(existing.fact_ids) | set(fact_ids)
            existing.fact_ids = list(new_ids)
            entities_to_save.append(existing)
            entity_ids.append(existing.id)
        else:
            eid = uuid.uuid4().hex[:12]
            entities_to_save.append(
                Entity(
                    id=eid,
                    bank_id=bank_id,
                    text=name,
                    entity_type=etype,
                    fact_ids=fact_ids,
                    created_at=now,
                )
            )
            entity_ids.append(eid)

    # Embed new entities (those without embeddings)
    to_embed = [e for e in entities_to_save if not e.embedding]
    if to_embed:
        try:
            embs = await embedder.embed([e.text for e in to_embed])
            for entity, emb in zip(to_embed, embs, strict=True):
                entity.embedding = emb
        except Exception:
            logger.exception("Entity embedding failed")

    store.save_entities(entities_to_save)
    return entity_ids
