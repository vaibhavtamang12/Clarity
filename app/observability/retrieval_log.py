"""RetrievalLog persistence (Phase 24, decision D-132).

The RetrievalLog model has existed since Phase 3 — this activates it.
Every RAG query leaves a durable audit row: query_id correlation, stage
latencies, retrieved chunk IDs, candidate/returned counts, cache hit flag.

Best-effort by contract: observability must never fail a request. Any write
failure logs a warning and moves on.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.generation.domain import RAGResponse
from app.models.logs import RetrievalLog

logger = get_logger(__name__)


class RetrievalLogWriter:
    async def write(
        self,
        session: AsyncSession,
        *,
        query_id: uuid.UUID,
        question: str,
        rewritten_query: str | None,
        response: RAGResponse,
        conversation_id: uuid.UUID | None = None,
        cache_hit: bool = False,
    ) -> None:
        try:
            log = RetrievalLog(
                query_id=query_id,
                conversation_id=conversation_id,
                message_id=None,
                query=question,
                rewritten_query=rewritten_query,
                retriever_config={
                    "retriever": response.retrieval.retriever,
                    "strategy": response.retrieval.strategy,
                    "num_items": response.retrieval.num_items,
                    "degraded": response.retrieval.degraded,
                    "counts": response.retrieval.counts,
                    "mode": response.mode,
                    "transform_type": response.transform_type,
                },
                retrieved_chunk_ids=[c.chunk_id for c in response.citations],
                scores=None,
                stage_latencies=response.stage_latencies_ms,
                num_candidates=response.retrieval.num_items,
                num_returned=len(response.citations),
                cache_hit=cache_hit,
            )
            session.add(log)
            await session.flush()
        except Exception as exc:  # noqa: BLE001 — observability never fails requests
            logger.warning("retrieval_log_write_failed", error=type(exc).__name__)