"""Conversation + Message repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select

from app.models.conversation import Conversation, Message
from app.models.enums import MessageRole
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

    async def create(self, user_id: uuid.UUID, title: str | None = None) -> Conversation:
        return await self.add(Conversation(user_id=user_id, title=title))

    async def get_owned(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation | None:
        """Ownership-checked fetch — the authorization boundary for conversations."""
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(
        self, user_id: uuid.UUID, limit: int = 20, offset: int = 0
    ) -> Sequence[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.last_message_at.desc().nulls_last())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def touch(self, conversation: Conversation) -> None:
        conversation.last_message_at = datetime.now(timezone.utc)
        await self.session.flush()


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def create(
        self,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        **fields: object,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            **fields,  # type: ignore[arg-type]
        )
        return await self.add(message)

    async def list_for_conversation(
        self, conversation_id: uuid.UUID, limit: int = 50
    ) -> Sequence[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()