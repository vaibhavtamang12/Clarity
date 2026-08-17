"""Retriever abstraction — the contract dense, sparse, hybrid, and (Phase 9)
reranked retrieval all implement.

Results carry provenance: which retriever(s) produced each chunk, raw branch
scores, and result-level metadata (strategy, degradation, per-stage latency).
This is the raw material for retrieval_logs (ARCHITECTURE §5.1) and the
Phase 23/24 observability layers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.repositories.vector.base import VectorFilter


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    score: float                      # fused score for hybrid; raw branch score otherwise
    content: str
    sources: tuple[str, ...] = ()     # e.g. ("dense",), ("sparse",), ("dense", "sparse")
    dense_score: float | None = None
    sparse_score: float | None = None
    section: str | None = None
    source_uri: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class RetrievalMetadata:
    retriever: str                                  # dense | sparse_postgres | sparse_bm25 | hybrid
    strategy: str | None = None                     # fusion strategy for hybrid
    degraded: bool = False
    degraded_reason: str | None = None
    branch_latencies_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    items: list[RetrievedChunk]
    metadata: RetrievalMetadata


@runtime_checkable
class Retriever(Protocol):
    name: str

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_: VectorFilter | None = None,
    ) -> RetrievalResult: ...

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    score: float                      # fused score (hybrid) or rerank score (reranked)
    content: str
    sources: tuple[str, ...] = ()
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None   # Phase 9 addition — set by RerankedRetriever
    section: str | None = None
    source_uri: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None