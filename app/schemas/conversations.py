# app/schemas/conversations.py
"""Conversation API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    status: str
    message_count: int = 0
    created_at: datetime
    last_message_at: datetime | None = None


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    limit: int
    offset: int
    has_more: bool


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    rewritten_query: str | None = None
    grounding_score: float | None = None
    latency_ms: int | None = None
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    conversation: ConversationSummary
    messages: list[MessageOut]