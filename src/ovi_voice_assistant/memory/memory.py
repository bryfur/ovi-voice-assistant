"""High-level Memory interface wrapping store + embedder + LLM."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import AsyncOpenAI

from ovi_voice_assistant.config import Settings
from ovi_voice_assistant.memory.embedder import Embedder
from ovi_voice_assistant.memory.recall import Budget, recall
from ovi_voice_assistant.memory.retain import retain
from ovi_voice_assistant.memory.store import MemoryStore
from ovi_voice_assistant.memory.types import RecallResult, RetainResult

logger = logging.getLogger(__name__)


class Memory:
    """High-level memory interface wrapping store + embedder + LLM.

    Usage::

        memory = Memory(settings)
        memory.load()
        result = await memory.retain("Alice is allergic to shellfish")
        facts = await memory.recall("What are Alice's allergies?")
        memory.close()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store: MemoryStore | None = None
        self._embedder: Embedder | None = None
        self._llm: AsyncOpenAI | None = None
        self._bank_id = settings.memory.bank_id

    @property
    def bank_id(self) -> str:
        return self._bank_id

    def load(self) -> None:
        """Open the SQLite store and load the local embedding model."""
        raw_path = self._settings.memory.db_path
        db_path = raw_path if raw_path == ":memory:" else Path(raw_path).expanduser()
        self._store = MemoryStore(db_path)
        self._store.open()

        self._embedder = Embedder(model=self._settings.memory.embedding_model)
        self._embedder.load()

        self._llm = AsyncOpenAI(
            base_url=self._settings.llm.base_url or "https://api.openai.com/v1",
            api_key=self._settings.llm.api_key or "not-set",
        )

        logger.info(
            "Memory loaded (db=%s, embedding=%s)",
            db_path,
            self._settings.memory.embedding_model,
        )

    def close(self) -> None:
        """Close the store."""
        if self._store:
            self._store.close()
            self._store = None
        logger.info("Memory closed")

    async def retain(self, content: str, context: str = "") -> RetainResult:
        """Extract and store facts from text."""
        if not self._store or not self._embedder or not self._llm:
            raise RuntimeError("Call load() first")
        return await retain(
            bank_id=self._bank_id,
            content=content,
            llm=self._llm,
            llm_model=self._settings.llm.model,
            embedder=self._embedder,
            store=self._store,
            context=context,
        )

    async def recall(
        self, query: str, budget: Budget = Budget.MID, max_tokens: int = 1024
    ) -> RecallResult:
        """Search memory for relevant facts."""
        if not self._store or not self._embedder:
            raise RuntimeError("Call load() first")
        return await recall(
            bank_id=self._bank_id,
            query=query,
            embedder=self._embedder,
            store=self._store,
            budget=budget,
            max_tokens=max_tokens,
        )
