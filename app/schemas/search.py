# app/schemas/search.py
"""Direct retrieval (no generation) API schemas."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: uuid.UUID | None = None
    version_id: uuid.UUID | None = None


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    score: float
    sources: list[str] = []
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    content: str
    section: str | None = None
    source_uri: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class SearchResponse(BaseModel):
    items: list[SearchHit]
    retriever: str
    strategy: str | None = None
    degraded: bool = False
    branch_latencies_ms: dict[str, float] = {}