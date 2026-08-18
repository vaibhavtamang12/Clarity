# app/api/v1/endpoints/chat.py
"""Chat endpoint — grounded answers with conversations (Phase 16)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ChatResponse:
    platform = get_platform(request)
    response, conversation_id = await platform.chat_service.answer(
        session=session,
        user_id=user.id,
        question=payload.question,
        conversation_id=payload.conversation_id,
    )
    return ChatResponse.from_rag(response, conversation_id)