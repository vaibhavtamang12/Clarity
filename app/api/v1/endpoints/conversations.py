# app/api/v1/endpoints/conversations.py
"""Conversation endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db
from app.conversation.storage import ConversationStorageService
from app.core.exceptions import ConversationNotFoundError
from app.repositories.conversation import ConversationRepository
from app.schemas.conversations import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummary,
    MessageOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _summary(conv) -> ConversationSummary:  # type: ignore[no-untyped-def]
    meta = conv.metadata_ or {}
    status = conv.status.value if hasattr(conv.status, "value") else str(conv.status)
    return ConversationSummary(
        id=conv.id, title=conv.title, status=status,
        message_count=int(meta.get("message_count", 0)),
        created_at=conv.created_at, last_message_at=conv.last_message_at,
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationListResponse:
    convs = await ConversationRepository(session).list_for_user(
        user.id, limit=limit + 1, offset=offset
    )
    return ConversationListResponse(
        items=[_summary(c) for c in convs[:limit]],
        limit=limit, offset=offset, has_more=len(convs) > limit,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationDetailResponse:
    storage = ConversationStorageService(session)
    conv = await storage.get_owned(conversation_id, user.id)  # 404 if foreign
    messages = await storage.list_messages(conversation_id, limit=200)
    return ConversationDetailResponse(
        conversation=_summary(conv),
        messages=[
            MessageOut(
                id=m.id,
                role=m.role.value if hasattr(m.role, "value") else str(m.role),
                content=m.content,
                rewritten_query=m.rewritten_query,
                grounding_score=m.grounding_score,
                latency_ms=m.latency_ms,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )