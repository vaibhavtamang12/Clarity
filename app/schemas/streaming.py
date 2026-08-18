"""SSE event schemas for streaming chat (Phase 17)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


class StatusEvent(BaseModel):
    """Operational status event (no internal reasoning exposed)."""
    
    stage: str                            # retrieving | reranking | generating | validating | regenerating | completed | error
    message: str | None = None


class TokenEvent(BaseModel):
    """Answer token (incremental text)."""
    
    token: str


class CitationEvent(BaseModel):
    """Citation metadata (emitted when generation completes)."""
    
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


class MetadataEvent(BaseModel):
    """Final metadata (grounding score, retrieval stats, etc.)."""
    
    query_id: uuid.UUID
    conversation_id: uuid.UUID
    mode: str
    confidence: float | None = None
    insufficient_evidence: bool = False
    grounding_score: float | None = None
    policy_action: str | None = None
    uncertainty_notice: str | None = None
    retrieval: dict[str, Any]
    stage_latencies_ms: dict[str, float]
    token_usage: dict[str, int] | None = None


class ErrorEvent(BaseModel):
    """Error event (stream closes after this)."""
    
    code: str
    message: str
    details: dict[str, Any] | None = None