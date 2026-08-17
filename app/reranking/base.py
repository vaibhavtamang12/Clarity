"""Reranker abstraction (ADR-004).

Same two-tier discipline as embeddings (D-034): the Reranker is synchronous
and pure — it scores (query, text) pairs and nothing else. Async wrapping,
timing, and failure policy live in RerankedRetriever (retriever.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.retrieval.base import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    model_key: str
    model_id: str

    def rerank(self, query: str, items: Sequence[RetrievedChunk]) -> list[float]:
        """Return relevance scores aligned 1:1 with `items` (higher = better)."""
        ...