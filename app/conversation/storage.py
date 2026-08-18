# app/conversation/storage.py
"""Conversation storage — maps RAG metadata onto the REAL message columns:
retrieval_metadata / citations / grounding_score / token_usage / latency_ms."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.domain import ConversationTurn
from app.core.exceptions import ConversationNotFoundError
from app.generation.domain import RAGResponse
from app.models.conversation import Conversation
from app.models.enums import MessageRole
from app.repositories.conversation import ConversationRepository, MessageRepository


class ConversationStorageService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)

    async def create_conversation(self, user_id: uuid.UUID, title: str | None = None) -> uuid.UUID:
        conv = await self._conversations.create(user_id=user_id, title=title)
        conv.metadata_ = {"message_count": 0}
        await self._session.flush()
        return conv.id

    async def get_owned(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
        conv = await self._conversations.get_owned(conversation_id, user_id)
        if conv is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        return conv

    async def add_user_message(self, conversation_id: uuid.UUID, content: str) -> None:
        await self._messages.create(
            conversation_id=conversation_id, role=MessageRole.USER, content=content
        )
        await self._bump(conversation_id)

    async def add_assistant_message(
        self, conversation_id: uuid.UUID, content: str, *, response: RAGResponse
    ) -> None:
        fields: dict = {
            "retrieval_metadata": {
                "retriever": response.retrieval.retriever,
                "strategy": response.retrieval.strategy,
                "num_items": response.retrieval.num_items,
                "branch_latencies_ms": response.retrieval.branch_latencies_ms,
                "counts": response.retrieval.counts,
                "degraded": response.retrieval.degraded,
                "mode": response.mode,
                "confidence": response.confidence,
                "insufficient_evidence": response.insufficient_evidence,
            },
            "citations": [c.model_dump(mode="json") for c in response.citations],
        }
        if response.retrieval_query != response.question:
            fields["rewritten_query"] = response.retrieval_query
        if response.grounding is not None:
            fields["grounding_score"] = response.grounding.score
        if response.token_usage is not None:
            fields["token_usage"] = response.token_usage.model_dump()
        total_ms = response.stage_latencies_ms.get("total_ms")
        if total_ms is not None:
            fields["latency_ms"] = int(total_ms)
        await self._messages.create(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            **fields,
        )
        await self._bump(conversation_id)

    async def list_messages(self, conversation_id: uuid.UUID, limit: int = 200):  # type: ignore[no-untyped-def]
        return await self._messages.list_for_conversation(conversation_id, limit=limit)

    async def get_turns(
        self, conversation_id: uuid.UUID, max_turns: int = 10
    ) -> list[ConversationTurn]:
        messages = await self._messages.list_for_conversation(
            conversation_id, limit=max_turns * 2
        )
        turns: list[ConversationTurn] = []
        i = 0
        while i < len(messages) - 1:
            user_msg, assistant_msg = messages[i], messages[i + 1]
            user_role = user_msg.role.value if hasattr(user_msg.role, "value") else str(user_msg.role)
            assistant_role = (
                assistant_msg.role.value if hasattr(assistant_msg.role, "value") else str(assistant_msg.role)
            )
            if user_role == "user" and assistant_role == "assistant":
                turns.append(
                    ConversationTurn(
                        user_message=user_msg.content,
                        assistant_message=assistant_msg.content,
                        turn_index=len(turns),
                        created_at=user_msg.created_at,
                        grounding_score=assistant_msg.grounding_score,
                    )
                )
                i += 2
            else:
                i += 1
        return turns

    async def _bump(self, conversation_id: uuid.UUID) -> None:
        conv = await self._conversations.get_by_id(conversation_id)
        if conv is None:
            return
        meta = dict(conv.metadata_ or {})
        meta["message_count"] = int(meta.get("message_count", 0)) + 1
        conv.metadata_ = meta
        await self._conversations.touch(conv)
        await self._session.flush()