"""Dense retriever — query embedding → filtered vector similarity.

Safety default (decision D-041): unless explicitly overridden, ONLY
active-version chunks are searched. Superseded content never leaks into
answers by accident.
"""

from __future__ import annotations

import time
from dataclasses import replace

from app.core.logging import get_logger
from app.embeddings.service import EmbeddingService
from app.repositories.vector.base import VectorFilter, VectorRepository
from app.retrieval.base import RetrievalMetadata, RetrievalResult, RetrievedChunk

logger = get_logger(__name__)


class DenseRetriever:
    name = "dense"

    def __init__(self, repository: VectorRepository, embedding_service: EmbeddingService) -> None:
        self._repository = repository
        self._embeddings = embedding_service

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_: VectorFilter | None = None,
        score_threshold: float | None = None,
        include_inactive: bool = False,
    ) -> RetrievalResult:
        start = time.perf_counter()
        vector = await self._embeddings.embed_query(query)

        vf = filter_ or VectorFilter()
        if vf.is_active_version is None and vf.version_id is None and not include_inactive:
            vf = replace(vf, is_active_version=True)

        results = await self._repository.search(
            vector, top_k, filter_=vf, score_threshold=score_threshold
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        items = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                version_id=r.version_id,
                score=r.score,
                content=r.content,
                sources=("dense",),
                dense_score=r.score,
                section=r.section,
                source_uri=r.source_uri,
                title=r.title,
                page_start=r.page_start,
                page_end=r.page_end,
            )
            for r in results
        ]
        logger.info(
            "dense_retrieval_completed",
            collection=self._repository.collection,
            top_k=top_k,
            returned=len(items),
            latency_ms=latency_ms,
        )
        return RetrievalResult(
            items=items,
            metadata=RetrievalMetadata(
                retriever=self.name,
                branch_latencies_ms={"dense": latency_ms},
                counts={"dense": len(items)},
            ),
        )