"""Sparse retrievers.

Two implementations, one interface (decision D-031 lineage):
- PostgresSparseRetriever — production path. Uses the computed tsvector
  column + GIN index created in Phase 3 with ts_rank_cd scoring.
- BM25Retriever — pure-Python Okapi BM25 over an in-memory chunk set,
  for benchmarks/experiments where infrastructure must stay out of the
  comparison (Phase 5 methodology).

Filter support differs and is EXPLICIT about it: the Postgres retriever
supports document/version/owner filters; anything it cannot enforce
correctly raises instead of silently ignoring (honest failure > silent drift).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select

from app.core.exceptions import RetrievalUnavailableError
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentVersion
from app.models.enums import VersionStatus
from app.repositories.database import Database
from app.repositories.vector.base import VectorFilter
from app.retrieval.base import RetrievalMetadata, RetrievalResult, RetrievedChunk
from app.retrieval.bm25 import BM25Index

logger = get_logger(__name__)

_TS_LANGUAGE = "english"

_UNSUPPORTED_SPARSE_FILTERS = (
    "source_type", "document_type", "department", "tags",
    "created_after", "created_before",
)


@dataclass(frozen=True)
class SparseChunkRecord:
    """Chunk identity for the in-memory BM25 retriever."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    content: str
    section: str | None = None
    source_uri: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class PostgresSparseRetriever:
    """Lexical retrieval over PostgreSQL full-text search (production path)."""

    name = "sparse_postgres"

    def __init__(self, database: Database) -> None:
        self._database = database

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_: VectorFilter | None = None,
    ) -> RetrievalResult:
        vf = filter_ or VectorFilter()
        for field_name in _UNSUPPORTED_SPARSE_FILTERS:
            if getattr(vf, field_name) is not None:
                raise ValueError(
                    f"PostgresSparseRetriever does not support filter '{field_name}' — "
                    "refusing to silently ignore a filter condition"
                )

        start = time.perf_counter()
        try:
            async with self._database.session() as session:
                tsq = func.plainto_tsquery(_TS_LANGUAGE, query)
                rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsq)

                stmt = (
                    select(DocumentChunk, rank.label("rank"))
                    .join(DocumentVersion, DocumentChunk.version_id == DocumentVersion.id)
                    .where(DocumentChunk.content_tsv.op("@@")(tsq))
                )
                if vf.version_id is not None:
                    # Explicit version query — history retrieval bypasses active-only.
                    stmt = stmt.where(DocumentChunk.version_id == vf.version_id)
                else:
                    stmt = stmt.where(DocumentVersion.status == VersionStatus.ACTIVE)
                if vf.document_id is not None:
                    stmt = stmt.where(DocumentChunk.document_id == vf.document_id)
                if vf.owner_id is not None:
                    stmt = stmt.join(Document, DocumentChunk.document_id == Document.id).where(
                        Document.owner_id == vf.owner_id
                    )
                stmt = stmt.order_by(rank.desc()).limit(top_k)

                rows = (await session.execute(stmt)).all()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 — typed failure at the boundary
            raise RetrievalUnavailableError(
                f"Sparse retrieval failed: {type(exc).__name__}"
            ) from exc

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        items = [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                score=float(score),
                content=chunk.content,
                sources=("sparse",),
                sparse_score=float(score),
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
            for chunk, score in rows
        ]
        logger.info(
            "sparse_retrieval_completed",
            top_k=top_k,
            returned=len(items),
            latency_ms=latency_ms,
        )
        return RetrievalResult(
            items=items,
            metadata=RetrievalMetadata(
                retriever=self.name,
                branch_latencies_ms={"sparse": latency_ms},
                counts={"sparse": len(items)},
            ),
        )


class BM25Retriever:
    """In-memory Okapi BM25 retriever for controlled experiments."""

    name = "sparse_bm25"

    def __init__(self, index: BM25Index, records: Sequence[SparseChunkRecord]) -> None:
        if index.document_count != len(records):
            raise ValueError("BM25 index and chunk records must be aligned")
        self._index = index
        self._records = list(records)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_: VectorFilter | None = None,
    ) -> RetrievalResult:
        if filter_ is not None and not filter_.is_empty():
            raise ValueError("BM25Retriever is corpus-scoped and accepts no metadata filters")
        start = time.perf_counter()
        scored = self._index.search(query, top_k)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        items = [
            RetrievedChunk(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                version_id=record.version_id,
                score=sd.score,
                content=record.content,
                sources=("sparse",),
                sparse_score=sd.score,
                section=record.section,
                source_uri=record.source_uri,
                page_start=record.page_start,
                page_end=record.page_end,
            )
            for sd in scored
            for record in [self._records[sd.index]]
        ]
        return RetrievalResult(
            items=items,
            metadata=RetrievalMetadata(
                retriever=self.name,
                branch_latencies_ms={"sparse": latency_ms},
                counts={"sparse": len(items)},
            ),
        )