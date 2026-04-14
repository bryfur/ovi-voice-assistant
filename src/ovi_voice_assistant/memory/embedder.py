"""Local embedding using fastembed (ONNX runtime, no torch)."""

from __future__ import annotations

import asyncio
import logging

from fastembed import TextEmbedding

from ovi_voice_assistant.config import CACHE_DIR

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings locally using a small ONNX model."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model: TextEmbedding | None = None

    def load(self) -> None:
        """Load the embedding model (downloads on first use)."""
        self._model = TextEmbedding(
            self._model_name, cache_dir=str(CACHE_DIR / "fastembed")
        )
        logger.info("Embedding model loaded: %s", self._model_name)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Call load() first")
        return [emb.tolist() for emb in self._model.embed(texts)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        results = await self.embed([text])
        return results[0] if results else []
