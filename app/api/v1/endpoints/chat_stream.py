"""Streaming chat endpoint (Phase 17)."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.schemas.chat import ChatRequest
from app.schemas.streaming import ErrorEvent

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""
    platform = get_platform(request)

    async def event_generator():
        try:
            response, conversation_id = await platform.chat_service.answer_stream(
                session=session,
                user_id=user.id,
                question=payload.question,
                conversation_id=payload.conversation_id,
            )

            async for chunk in response:
                # Format as SSE: event: <type>\ndata: <json>\n\n
                event_type = chunk.event_type
                data = chunk.data.model_dump_json()
                yield f"event: {event_type}\ndata: {data}\n\n"

        except Exception as exc:
            # Stream error event
            error = ErrorEvent(
                code="STREAM_ERROR",
                message=str(exc),
            )
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )