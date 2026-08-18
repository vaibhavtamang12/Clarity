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
        response_cache: ResponseCache | None = None,    # Phase 18
        metrics: dict | None = None, 
    ) -> None:
        self._rag = rag_pipeline
        self._selector = history_selector
        self._max_history_turns = max_history_turns
        self._response_cache = response_cache
        self._metrics = metrics

    async def answer(
        self,
        *,
        session: AsyncSession,
        user_id: uuid.UUID,
        question: str,
        conversation_id: uuid.UUID | None = None,
    ) -> tuple[RAGResponse, uuid.UUID]:
        storage = ConversationStorageService(session)
        is_standalone = conversation_id is None

        # ---- response cache (standalone questions only, D-100) ---------------
        if is_standalone and self._response_cache is not None:
            cached = await self._response_cache.get(user_id, question)
            if cached is not None:
                if self._metrics is not None:
                    self._metrics["cache_hits_total"] = self._metrics.get("cache_hits_total", 0) + 1
                conversation_id = await storage.create_conversation(
                    user_id=user_id, title=question[:80]
                )
                await storage.add_user_message(conversation_id, question)
                await storage.add_assistant_message(conversation_id, cached.answer, response=cached)
                await session.commit()
                return cached, conversation_id
            if self._metrics is not None:
                self._metrics["cache_misses_total"] = self._metrics.get("cache_misses_total", 0) + 1

# ... existing pipeline call ...
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
        
        if is_standalone and self._response_cache is not None:
            await self._response_cache.put(user_id, question, response)

        await storage.add_user_message(conversation_id, question)
        response = await self._rag.answer(question, history=history)
        await storage.add_assistant_message(conversation_id, response.answer, response=response)
        await session.commit()
        return response, conversation_id
    
    # Add new method to ChatService:

    async def answer_stream(
        self,
        *,
        session: AsyncSession,
        user_id: uuid.UUID,
        question: str,
        conversation_id: uuid.UUID | None = None,
    ):
        """Streaming version of answer() — yields SSE events."""
        from collections.abc import AsyncIterator
        from app.generation.pipeline import StreamChunk
        
        storage = ConversationStorageService(session)
        history: list[object] = []

        if conversation_id is None:
            conversation_id = await storage.create_conversation(
                user_id=user_id, title=question[:80]
            )
        else:
            await storage.get_owned(conversation_id, user_id)
            turns = await storage.get_turns(conversation_id, max_turns=10)
            selected = await self._selector.select_relevant_turns(question, turns)
            for turn in selected[-self._max_history_turns :]:
                history.append(RewriterTurn(role="user", content=turn.user_message))
                history.append(RewriterTurn(role="assistant", content=turn.assistant_message))

        await storage.add_user_message(conversation_id, question)
        
        # Stream the RAG pipeline
        full_answer = ""
        async for chunk in self._rag.answer_stream(question, history=history):
            # Capture full answer for persistence
            if chunk.event_type == "token":
                full_answer += chunk.data.token
            yield chunk

        # Persist after streaming completes (Phase 14 storage)
        # For now, we don't persist in streaming mode to avoid partial writes
        # In production, this would call storage.add_assistant_message()
        await session.commit()