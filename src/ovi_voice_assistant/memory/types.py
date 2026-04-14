"""Data types for the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FactType(StrEnum):
    """What kind of fact this is."""

    WORLD = "world"  # Objective info about the world
    EXPERIENCE = "experience"  # User's experiences/preferences
    ASSISTANT = "assistant"  # Things the assistant did/said


class EntityType(StrEnum):
    """Classification for extracted entities."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    OTHER = "other"


@dataclass
class Fact:
    """A single extracted fact stored in memory."""

    id: str
    bank_id: str
    text: str  # Combined fact text for embedding
    what: str  # Core fact (1-2 sentences)
    who: str = ""
    where: str = ""
    when: str = ""
    why: str = ""
    fact_type: FactType = FactType.WORLD
    confidence: float = 1.0
    embedding: list[float] = field(default_factory=list)
    created_at: str = ""  # ISO timestamp
    occurred_at: str = ""  # When the fact's event occurred (ISO)
    relevance: float = 0.0  # Set during recall, not stored


@dataclass
class Entity:
    """A named entity linked to facts."""

    id: str
    bank_id: str
    text: str  # Canonical name
    entity_type: EntityType = EntityType.OTHER
    embedding: list[float] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class RetainResult:
    """Result from a retain operation."""

    success: bool
    facts_count: int
    fact_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class RecallResult:
    """Result from a recall operation."""

    results: list[Fact]
    entities: list[Entity] = field(default_factory=list)
    total_candidates: int = 0
