"""Embedding abstraction (ADR-002, PROJECT_SPEC Phase 6).

Two-tier design (decision D-034):
- EmbeddingModel: synchronous, pure ML. No I/O concerns. Trivially testable.
- EmbeddingService (service.py): asynchronous operational wrapper — batching,
  caching, retries, timing. Runs the sync model via asyncio.to_thread.

Models are configuration, not code (Rule 3): everything identity-related
(key, id, version, dimension) comes from configs/embeddings.yaml.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """A model that turns text into fixed-size vectors."""

    model_key: str       # registry key, e.g. "bge_m3"
    model_id: str        # upstream identifier, e.g. "BAAI/bge-m3"
    model_version: str   # version stamp stored with every vector produced
    dimension: int
    max_tokens: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunk/document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (may apply a model-specific query prefix)."""
        ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk vector ready for the vector index (consumed by Phase 7).

    point_id is the deterministic UUIDv5 minted at ingestion time, so the
    same chunk always maps to the same Qdrant point (idempotent upserts).
    """

    chunk_id: uuid.UUID
    point_id: uuid.UUID
    vector: list[float]
    token_count: int