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
    # ... all Phase 11 fields unchanged ...
    citation_validation: CitationValidation | None = None   # Phase 12 addition

# ---- Phase 12: citation validation models ---------------------------------


class CitationVerdict(StrEnum):
    CLEAN = "clean"        # every claim supported, no hygiene issues
    FLAGGED = "flagged"    # some issues found; answer still usable with caveats
    POOR = "poor"          # majority of claims unsupported — Phase 13 policy input


class ClaimVerification(BaseModel):
    claim: str
    markers: list[int] = Field(default_factory=list)
    status: str                       # supported | partial | unsupported
    best_score: float
    unmarked: bool = False            # claim sentence carries no citation at all


class CitationCheck(BaseModel):
    passage: int
    chunk_id: uuid.UUID
    declared_claim: str
    status: str                       # supported | partial | unsupported
    score: float


class CitationValidation(BaseModel):
    verdict: CitationVerdict
    final_citations: list["Citation"] = Field(default_factory=list)
    claims: list[ClaimVerification] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    claims_total: int = 0
    claims_supported: int = 0
    claims_partial: int = 0
    claims_unsupported: int = 0
    claim_support_rate: float = 1.0
    dangling_markers: list[int] = Field(default_factory=list)
    unused_citations: list[int] = Field(default_factory=list)
    markers_present: bool = False
    citation_correctness: float = 1.0   # declared claims verified / declared claims
    summary: str = ""

class GroundingMetrics(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    total: int = 0
    judge_adjudications: int = 0
    judge_failures: int = 0


class GroundingPolicyDecision(BaseModel):
    action: str                        # accept | regenerate | retrieve_more | uncertainty
    reason: str
    regeneration_attempts: int = 0
    context_expanded: bool = False
    uncertainty_notice: str | None = None


class RAGResponse(BaseModel):
    # ... all Phase 11/12 fields unchanged ...
    grounding: GroundingMetrics | None = None               # Phase 13 addition
    grounding_policy: GroundingPolicyDecision | None = None  # Phase 13 addition