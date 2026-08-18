"""Conversation domain types (Phase 14).

Conversation metadata captures the full RAG context: what was retrieved, what
was generated, what citations were used, what the grounding score was. This
enables conversation-aware retrieval (what was already covered) and selective
history injection (what's relevant to the current query).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConversationMetadata(BaseModel):
    """Conversation-level metadata."""

    title: str | None = None
    purpose: str | None = None              # e.g., "research", "support", "general"
    topic: str | None = None                # e.g., "refund policy", "security"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    total_tokens: int = 0                   # cumulative token usage
    average_grounding_score: float | None = None


class MessageRetrievalMetadata(BaseModel):
    """What was retrieved for this message."""

    retriever: str                          # e.g., "hybrid_rrf", "dense", "sparse"
    strategy: str | None = None             # e.g., "rrf", "weighted"
    num_items: int = 0
    branch_latencies_ms: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str | None = None


class MessageCitation(BaseModel):
    """A citation used in this message."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    passage: int
    claim: str = ""
    page_start: int | None = None
    section: str | None = None
    source_uri: str | None = None


class MessageMetadata(BaseModel):
    """Message-level RAG metadata."""

    role: Literal["user", "assistant"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Retrieval metadata (assistant messages only)
    retrieval: MessageRetrievalMetadata | None = None
    # Citations (assistant messages only)
    citations: list[MessageCitation] = Field(default_factory=list)
    # Grounding (assistant messages only)
    grounding_score: float | None = None
    grounding_policy_action: str | None = None  # e.g., "accept", "regenerate", "uncertainty"
    # Token usage
    token_usage: dict[str, int] = Field(default_factory=dict)  # prompt, completion, total
    # Stage latencies
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    # Query transformation
    transform_type: str | None = None       # e.g., "none", "rewrite", "expansion"
    rewritten_query: str | None = None
    # Context
    context_tokens: int = 0
    # Generation
    mode: str | None = None                 # e.g., "generated", "retrieval_only"
    confidence: float | None = None
    insufficient_evidence: bool = False


@dataclass(frozen=True)
class ConversationTurn:
    user_message: str
    assistant_message: str
    turn_index: int
    created_at: datetime | None = None
    grounding_score:

@dataclass
class ConversationContext:
    """Context for the current conversation (passed to RAG pipeline)."""

    conversation_id: uuid.UUID
    turns: list[ConversationTurn]
    metadata: ConversationMetadata