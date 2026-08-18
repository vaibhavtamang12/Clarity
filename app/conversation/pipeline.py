"""Conversation-aware RAG pipeline (Phase 14).

Wraps the base RAGPipeline with conversation memory:
1. Retrieves conversation context
2. Selects relevant history turns
3. Injects history into query transformation (Phase 10)
4. Stores the response with full metadata
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.conversation.domain import ConversationContext, ConversationTurn
from app.conversation.history import HistorySelector
from app.conversation.retriever import ConversationalRetriever
from app.conversation.storage import ConversationStorageService
from app.generation.domain import RAGResponse
from app.generation.pipeline import RAGPipeline
from app.repositories.vector.base import VectorFilter


class ConversationalRAGPipeline:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        storage: ConversationStorageService,
        history_selector: HistorySelector,
        conversational_retriever: ConversationalRetriever | None = None,
    ) -> None:
        self._rag = rag_pipeline
        self._storage = storage
        self._selector = history_selector
        self._retriever = conversational_retriever

    async def answer(
        self,
        question: str,
        conversation_id: uuid.UUID | None = None,
        filter_: VectorFilter | None = None,
    ) -> RAGResponse:
        """Answer a question with conversation memory."""
        # Create conversation if needed
        if conversation_id is None:
            # This would require a user_id — for now, skip conversation creation
            conversation_context = None
        else:
            # Retrieve conversation context
            conversation_context = await self._storage.get_conversation_context(
                conversation_id, max_turns=10
            )

        # Select relevant history turns
        history: list[ConversationTurn] = []
        if conversation_context is not None:
            history = await self._selector.select_relevant_turns(question, conversation_context)

        # Store user message
        if conversation_id is not None:
            from app.conversation.domain import MessageMetadata
            await self._storage.add_user_message(
                conversation_id, question, MessageMetadata(role="user")
            )

        # Generate response (pass history to query transformation)
        response = await self._rag.answer(
            question,
            history=history,
            filter_=filter_,
        )

        # Store assistant message with full metadata
        if conversation_id is not None:
            from app.conversation.domain import (
                MessageCitation,
                MessageMetadata,
                MessageRetrievalMetadata,
            )
            citations = [
                MessageCitation(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    passage=c.passage,
                    claim=c.claim,
                    page_start=c.page_start,
                    section=c.section,
                    source_uri=c.source_uri,
                )
                for c in response.citations
            ]
            retrieval_meta = MessageRetrievalMetadata(
                retriever=response.retrieval.retriever,
                strategy=response.retrieval.strategy,
                num_items=response.retrieval.num_items,
                branch_latencies_ms=response.retrieval.branch_latencies_ms,
                counts=response.retrieval.counts,
                degraded=response.retrieval.degraded,
                degraded_reason=response.retrieval.degraded_reason,
            )
            assistant_meta = MessageMetadata(
                role="assistant",
                retrieval=retrieval_meta,
                citations=citations,
                grounding_score=response.grounding.score if response.grounding else None,
                grounding_policy_action=response.grounding_policy.action if response.grounding_policy else None,
                token_usage=response.token_usage.model_dump() if response.token_usage else {},
                stage_latencies_ms=response.stage_latencies_ms,
                transform_type=response.transform_type,
                rewritten_query=response.retrieval_query if response.retrieval_query != response.question else None,
                context_tokens=sum(1 for _ in response.citations),  # approximate
                mode=response.mode,
                confidence=response.confidence,
                insufficient_evidence=response.insufficient_evidence,
            )
            await self._storage.add_assistant_message(
                conversation_id, response.answer, assistant_meta
            )

        return response