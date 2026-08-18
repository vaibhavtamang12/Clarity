# app/schemas/chat.py
"""Chat API schemas — structured response contract for RAG answers."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.generation.domain import RAGResponse


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None


class CitationOut(BaseModel):
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


class GroundingOut(BaseModel):
    score: float
    supported: int
    partial: int
    unsupported: int
    total: int
    judge_adjudications: int = 0


class RetrievalOut(BaseModel):
    retriever: str
    strategy: str | None = None
    num_items: int = 0
    degraded: bool = False
    degraded_reason: str | None = None


class ChatResponse(BaseModel):
    query_id: uuid.UUID
    conversation_id: uuid.UUID
    mode: str
    answer: str
    citations: list[CitationOut] = []
    confidence: float | None = None
    insufficient_evidence: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    transform_type: str
    grounding: GroundingOut | None = None
    policy_action: str | None = None
    uncertainty_notice: str | None = None
    retrieval: RetrievalOut
    stage_latencies_ms: dict[str, float] = {}
    token_usage: dict[str, int] | None = None

    @classmethod
    def from_rag(cls, response: RAGResponse, conversation_id: uuid.UUID) -> "ChatResponse":
        return cls(
            query_id=response.query_id,
            conversation_id=conversation_id,
            mode=response.mode,
            answer=response.answer,
            citations=[CitationOut(**c.model_dump()) for c in response.citations],
            confidence=response.confidence,
            insufficient_evidence=response.insufficient_evidence,
            degraded=response.degraded,
            degraded_reason=response.degraded_reason,
            transform_type=response.transform_type,
            grounding=GroundingOut(**response.grounding.model_dump()) if response.grounding else None,
            policy_action=response.grounding_policy.action if response.grounding_policy else None,
            uncertainty_notice=(
                response.grounding_policy.uncertainty_notice
                if response.grounding_policy
                else None
            ),
            retrieval=RetrievalOut(
                retriever=response.retrieval.retriever,
                strategy=response.retrieval.strategy,
                num_items=response.retrieval.num_items,
                degraded=response.retrieval.degraded,
                degraded_reason=response.retrieval.degraded_reason,
            ),
            stage_latencies_ms=response.stage_latencies_ms,
            token_usage=response.token_usage.model_dump() if response.token_usage else None,
        )