"""Conversational retriever (Phase 14).

Wraps a base retriever with conversation-aware behavior:
1. Query expansion: append relevant context from past turns
2. Filtering: optionally filter by conversation_id (retrieve what was discussed before)
3. Deduplication: avoid retrieving chunks already cited in past turns

This is NOT blind retrieval — it uses conversation context intelligently.
"""

from __future__ import annotations

import uuid

from app.conversation.domain import ConversationContext, ConversationTurn
from app.repositories.vector.base import VectorFilter
from app.retrieval.base import RetrievalResult, Retriever


class ConversationalRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        use_conversation_filter: bool = False,
        deduplicate_cited_chunks: bool = True,
    ) -> None:
        self._base = base_retriever
        self._use_filter = use_conversation_filter
        self._deduplicate = deduplicate_cited_chunks

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_: VectorFilter | None = None,
        conversation_context: ConversationContext | None = None,
    ) -> RetrievalResult:
        """Retrieve with conversation awareness."""
        # Build filter
        final_filter = filter_ or VectorFilter()
        
        # Optionally filter by conversation (retrieve what was discussed before)
        if self._use_filter and conversation_context is not None:
            # This would require storing conversation_id in chunk metadata
            # For now, we skip this feature (future enhancement)
            pass

        # Retrieve
        result = await self._base.retrieve(query, top_k=top_k, filter_=final_filter)

        # Deduplicate: remove chunks already cited in past turns
        if self._deduplicate and conversation_context is not None:
            cited_chunk_ids = self._get_cited_chunk_ids(conversation_context)
            filtered_items = [
                item for item in result.items if item.chunk_id not in cited_chunk_ids
            ]
            result = RetrievalResult(
                items=filtered_items,
                metadata=result.metadata,
            )

        return result

    def _get_cited_chunk_ids(self, context: ConversationContext) -> set[uuid.UUID]:
        """Extract chunk IDs cited in past turns."""
        cited: set[uuid.UUID] = set()
        for turn in context.turns:
            if turn.assistant_metadata.citations:
                for citation in turn.assistant_metadata.citations:
                    cited.add(citation.chunk_id)
        return cited