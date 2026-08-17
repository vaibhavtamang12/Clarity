"""Generation domain types — the structured contract of a RAG answer.

The target schema from PROJECT_SPEC Phase 11, extended with everything the
citation engine (12), grounding detector (13), and evaluation layers (21-22)
need: provenance per citation, stage latencies, degradation flags, mode.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.base import TokenUsage


class LLMCitationRef(BaseModel):
    """What the LLM emits: a passage NUMBER plus the claim it supports.
    The model never knows chunk IDs — fabrication is structurally impossible (D-058)."""

    passage: int = Field(ge=1)
    claim: str = ""


class LLMGenerationOutput(BaseModel):
    """Strict schema enforced on every generation attempt."""

    answer: str = Field(min_length=1, max_length=4000)
    citations: list[LLMCitationRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False


class Citation(BaseModel):
    """A resolved, verifiable citation (spec Phase 11/12 fields)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    passage: int
    claim: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source_uri: str | None = None
    title: str | None = None


class RetrievalStats(BaseModel):
    retriever: str
    strategy: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    num_items: int = 0
    branch_latencies_ms: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """The complete answer object returned to callers (API in Phase 16)."""

    query_id: uuid.UUID
    question: str
    retrieval_query: str
    transform_type: str
    mode: Literal["generated", "retrieval_only"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None   # self-reported by the LLM (D-062)
    insufficient_evidence: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    retrieval: RetrievalStats
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None