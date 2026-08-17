"""RerankedRetriever — the decorator that realizes ADR-004's pipeline:

    base retriever (top-50) → cross-encoder → top-10

Decorates ANY Retriever (dense, sparse, hybrid) and is itself a Retriever,
so it composes anywhere (D-048). Failure policy (D-049): if the reranker
fails AFTER retrieval succeeded, we serve the unreranked base ordering
flagged as degraded — a good top-10 beats a 503 at this stage.
"""

from __future__ import annotations

import asyncio
import time

from app.core.logging import get_logger
from app.reranking.base import Reranker
from app.repositories.vector.base import VectorFilter
from app.retrieval.base import RetrievalMetadata, RetrievalResult, RetrievedChunk, Retriever

logger = get_logger(__name__)


class RerankedRetriever:
    name = "reranked"

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        candidates: int = 50,
        top_n: int = 10,
        enabled: bool = True,
    ) -> None:
        if candidates < top_n:
            raise ValueError("candidates must be >= top_n")
        self._base = base
        self._reranker = reranker
        self._candidates = candidates
        self._top_n = top_n
        self._enabled = enabled

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter_: VectorFilter | None = None,
    ) -> RetrievalResult:
        final_k = top_k or self._top_n

        if not self._enabled:
            return await self._base.retrieve(query, top_k=final_k, filter_=filter_)

        # ---- retrieve wide --------------------------------------------------
        base_result = await self._base.retrieve(
            query, top_k=self._candidates, filter_=filter_
        )

        # ---- rerank narrow (in a thread — scoring is CPU-bound) -------------
        start = time.perf_counter()
        try:
            scores = await asyncio.to_thread(
                self._reranker.rerank, query, base_result.items
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft after successful retrieval
            logger.warning("rerank_failed_serving_unreranked", error=type(exc).__name__)
            return RetrievalResult(
                items=base_result.items[:final_k],
                metadata=RetrievalMetadata(
                    retriever=self.name,
                    strategy=f"{base_result.metadata.strategy or base_result.metadata.retriever}+rerank_skipped",
                    degraded=True,
                    degraded_reason=f"rerank failed: {type(exc).__name__}",
                    branch_latencies_ms={**base_result.metadata.branch_latencies_ms},
                    counts={
                        **base_result.metadata.counts,
                        "rerank_candidates": len(base_result.items),
                        "rerank_kept": min(final_k, len(base_result.items)),
                    },
                ),
            )
        rerank_ms = round((time.perf_counter() - start) * 1000, 2)

        if len(scores) != len(base_result.items):
            raise ValueError("Reranker returned misaligned scores")

        # ---- order by rerank score; stable tiebreak on base order -----------
        ranked = sorted(
            zip(base_result.items, scores, range(len(scores))),
            key=lambda entry: (-entry[1], entry[2]),
        )
        items = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                version_id=item.version_id,
                score=score,
                content=item.content,
                sources=item.sources,
                dense_score=item.dense_score,
                sparse_score=item.sparse_score,
                rerank_score=score,
                section=item.section,
                source_uri=item.source_uri,
                title=item.title,
                page_start=item.page_start,
                page_end=item.page_end,
            )
            for item, score, _pos in ranked[:final_k]
        ]

        logger.info(
            "rerank_completed",
            model=self._reranker.model_key,
            candidates=len(base_result.items),
            kept=len(items),
            latency_ms=rerank_ms,
        )
        return RetrievalResult(
            items=items,
            metadata=RetrievalMetadata(
                retriever=self.name,
                strategy=f"{base_result.metadata.strategy or base_result.metadata.retriever}+rerank",
                degraded=base_result.metadata.degraded,
                degraded_reason=base_result.metadata.degraded_reason,
                branch_latencies_ms={
                    **base_result.metadata.branch_latencies_ms,
                    "rerank": rerank_ms,
                },
                counts={
                    **base_result.metadata.counts,
                    "rerank_candidates": len(base_result.items),
                    "rerank_kept": len(items),
                },
            ),
        )