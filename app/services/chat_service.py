"""ChatService — conversation-aware orchestration for the /chat endpoint.

Phase 24 additions: every completed answer is instrumented (Prometheus)
and persisted to retrieval_logs (durable audit), both best-effort.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.history import HistorySelector
from app.conversation.storage import ConversationStorageService
from app.generation.pipeline import RAGPipeline
from app.observability.instrumentation import instrument_cache_event, instrument_rag_response
from app.observability.logging_policy import bind_query_context, clear_query_context
from app.observability.retrieval_log import RetrievalLogWriter
from app.retrieval.query.domain import ConversationTurn as RewriterTurn
from app.services.response_cache import ResponseCache


class ChatService:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        history_selector: HistorySelector,
        max_history_turns: int = 5,
        response_cache: ResponseCache | None = None,
        metrics: dict | None = None,
        retrieval_log_writer: RetrievalLogWriter | None = None,   # Phase 24
        llm_provider_name: str = "llm",                            # Phase 24
    ) -> None:
        self._rag = rag_pipeline
        self._selector = history_selector
        self._max_history_turns = max_history_turns
        self._response_cache = response_cache
        self._metrics = metrics
        self._log_writer = retrieval_log_writer or RetrievalLogWriter()
        self._llm_provider_name = llm_provider_name

    async def answer(
        self,
        *,
        session: AsyncSession,
        user_id: uuid.UUID,
        question: str,
        conversation_id: uuid.UUID | None = None,
    ) -> tuple[object, uuid.UUID]:
        storage = ConversationStorageService(session)
        history: list[object] = []
        is_standalone = conversation_id is None
        cache_hit = False

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

        # ---- response cache (standalone questions only, D-100) -----------------
        if is_standalone and self._response_cache is not None:
            cached = await self._response_cache.get(user_id, question)
            if cached is not None:
                cache_hit = True
                instrument_cache_event("response", "hit")
                if self._metrics is not None:
                    self._metrics["cache_hits_total"] = self._metrics.get("cache_hits_total", 0) + 1
                await storage.add_user_message(conversation_id, question)
                await storage.add_assistant_message(conversation_id, cached.answer, response=cached)
                await self._log_writer.write(
                    session,
                    query_id=cached.query_id,
                    question=question,
                    rewritten_query=None,
                    response=cached,
                    conversation_id=conversation_id,
                    cache_hit=True,
                )
                await session.commit()
                return cached, conversation_id
            instrument_cache_event("response", "miss")
            if self._metrics is not None:
                self._metrics["cache_misses_total"] = self._metrics.get("cache_misses_total", 0) + 1

        # ---- run the pipeline -----------------------------------------------------
        await storage.add_user_message(conversation_id, question)
        response = await self._rag.answer(question, history=history)

        bind_query_context(
            query_id=str(response.query_id), conversation_id=str(conversation_id)
        )
        try:
            # ---- observability: metrics + durable audit log (best-effort) ----------
            instrument_rag_response(response, llm_provider=self._llm_provider_name)
            await self._log_writer.write(
                session,
                query_id=response.query_id,
                question=question,
                rewritten_query=(
                    response.retrieval_query
                    if response.retrieval_query != question
                    else None
                ),
                response=response,
                conversation_id=conversation_id,
                cache_hit=cache_hit,
            )

            # ---- persist + cache -----------------------------------------------------
            await storage.add_assistant_message(
                conversation_id, response.answer, response=response
            )
            if is_standalone and self._response_cache is not None:
                await self._response_cache.put(user_id, question, response)
            await session.commit()
        finally:
            clear_query_context()

        return response, conversation_id