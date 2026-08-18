"""ChatService — conversation-aware orchestration for the /chat endpoint.

Flow (decision D-088):
1. Resolve or create the conversation (ownership enforced)
2. Select RELEVANT history turns (Phase 14 selector — never blind injection)
3. Convert selected turns into rewriter-format history (Phase 10)
4. Run the RAG pipeline
5. Persist both messages with full RAG metadata (schema-correct, D-089)
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.history import HistorySelector
from app.conversation.storage import ConversationStorageService
from app.generation.domain import RAGResponse
from app.generation.pipeline import RAGPipeline
from app.retrieval.query.domain import ConversationTurn as RewriterTurn


class ChatService:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        history_selector: HistorySelector,
        max_history_turns: int = 5,
    ) -> None:
        self._rag = rag_pipeline
        self._selector = history_selector
        self._max_history_turns = max_history_turns

    async def answer(
        self,
        *,
        session: AsyncSession,
        user_id: uuid.UUID,
        question: str,
        conversation_id: uuid.UUID | None = None,
    ) -> tuple[RAGResponse, uuid.UUID]:
        storage = ConversationStorageService(session)
        history: list[object] = []

        if conversation_id is None:
            conversation_id = await storage.create_conversation(
                user_id=user_id, title=question[:80]
            )
        else:
            await storage.get_owned(conversation_id, user_id)  # 404 if foreign
            turns = await storage.get_turns(conversation_id, max_turns=10)
            selected = await self._selector.select_relevant_turns(question, turns)
            for turn in selected[-self._max_history_turns :]:
                history.append(RewriterTurn(role="user", content=turn.user_message))
                history.append(RewriterTurn(role="assistant", content=turn.assistant_message))

        await storage.add_user_message(conversation_id, question)
        response = await self._rag.answer(question, history=history)
        await storage.add_assistant_message(conversation_id, response.answer, response=response)
        await session.commit()
        return response, conversation_id