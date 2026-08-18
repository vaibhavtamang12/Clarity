"""RAGPipeline — the full query path from PROJECT_SPEC Phase 11:

    transform → hybrid retrieval (+rerank) → context → generate → resolve
    → validated RAGResponse

Decomposition fan-out (D-060): sub-questions run through the FULL retriever
stack (including reranking) individually; results merge by chunk_id keeping
the best score. Merged-then-rerank is the alternative; it is compared in
Phase 22, not assumed.

Failure behavior (ARCHITECTURE §9): LLM unavailable → retrieval-only mode
with top passages as citations. Degradation is metadata, never silence.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence

from app.core.config import GenerationSettings
from app.core.logging import get_logger
from app.generation.citations import resolve_citations
from app.generation.context import ContextBuilder
from app.generation.domain import Citation, RAGResponse, RetrievalStats
from app.generation.generator import RAGGenerator
from app.repositories.vector.base import VectorFilter
from app.generation.engine import CitationEngine
from app.retrieval.base import RetrievalResult, RetrievedChunk, Retriever
from app.retrieval.query.domain import TransformedQuery, TransformType
from app.retrieval.query.service import QueryTransformService
from app.generation.adjudicator import ClaimAdjudicator
from app.generation.grounding import (
    GroundingScore,
    PolicyAction,
    compute_grounding_score,
    decide_action,
)
from app.generation.prompts import build_strict_user_message


logger = get_logger(__name__)


def _merge_items(result_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
    best: dict[uuid.UUID, RetrievedChunk] = {}
    for items in result_lists:
        for item in items:
            current = best.get(item.chunk_id)
            if current is None or item.score > current.score:
                best[item.chunk_id] = item
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: RAGGenerator,
        context_builder: ContextBuilder,
        settings: GenerationSettings,
        transform_service: QueryTransformService | None = None,
        citation_engine: CitationEngine | None = None,
        adjudicator: ClaimAdjudicator | None = None,      # Phase 13 addition
        grounding_settings: GroundingSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._context_builder = context_builder
        self._settings = settings
        self._transforms = transform_service
        ...
        self._citation_engine = citation_engine or CitationEngine()
        self._adjudicator = adjudicator
        self._grounding_settings = grounding_settings or GroundingSettings()

    async def answer(
        self,
        question: str,
        history: Sequence[object] = (),
        filter_: VectorFilter | None = None,
    ) -> RAGResponse:
        query_id = uuid.uuid4()
        stages: dict[str, float] = {}

        # ---- 1. query transformation (optional) ------------------------------
        start = time.perf_counter()
        if self._transforms is not None:
            transformed = await self._transforms.transform(question, history)
        else:
            transformed = TransformedQuery(
                original=question, retrieval_query=question,
                transform_type=TransformType.NONE, reasons=("transform service not wired",),
            )
        stages["transform_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # ---- 2. retrieval (+rerank inside the retriever) ----------------------
        start = time.perf_counter()
        if transformed.sub_queries:
            sub_results = await asyncio.gather(
                *(
                    self._retriever.retrieve(sub_q, filter_=filter_)
                    for sub_q in transformed.sub_queries
                )
            )
            retrieval = RetrievalResult(
                items=_merge_items([r.items for r in sub_results]),
                metadata=sub_results[0].metadata,
            )
        else:
            retrieval = await self._retriever.retrieve(
                transformed.retrieval_query, filter_=filter_
            )
        stages["retrieval_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # ---- 3. context selection ---------------------------------------------
        start = time.perf_counter()
        pack = self._context_builder.build(retrieval.items, self._settings.max_context_tokens)
        stages["context_ms"] = round((time.perf_counter() - start) * 1000, 2)

        retrieval_stats = RetrievalStats(
            retriever=retrieval.metadata.retriever,
            strategy=retrieval.metadata.strategy,
            degraded=retrieval.metadata.degraded,
            degraded_reason=retrieval.metadata.degraded_reason,
            num_items=len(retrieval.items),
            branch_latencies_ms=dict(retrieval.metadata.branch_latencies_ms),
            counts=dict(retrieval.metadata.counts),
        )

        # ---- 4. generation -----------------------------------------------------
        start = time.perf_counter()
        outcome = await self._generator.generate(question, pack)
        stages["generation_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # ---- 5a. fallback: retrieval-only mode (D-059) --------------------------
        if outcome.output is None:
            fallback_citations = [
                Citation(
                    chunk_id=p.chunk.chunk_id, document_id=p.chunk.document_id,
                    version_id=p.chunk.version_id, passage=p.number,
                    page_start=p.chunk.page_start, page_end=p.chunk.page_end,
                    section=p.chunk.section, source_uri=p.chunk.source_uri,
                    title=p.chunk.title,
                )
                for p in pack.passages[: self._settings.max_citations]
            ]
            logger.warning("rag_pipeline_fallback_retrieval_only", error=outcome.error)
            return RAGResponse(
                query_id=query_id, question=question,
                retrieval_query=transformed.retrieval_query,
                transform_type=transformed.transform_type.value,
                mode="retrieval_only",
                answer=self._settings.fallback_notice,
                citations=fallback_citations,
                confidence=None,
                degraded=True,
                degraded_reason=f"generation unavailable: {outcome.error}",
                retrieval=retrieval_stats,
                stage_latencies_ms=stages,
                token_usage=outcome.usage,
            )

        # ---- 5b. citation resolution + validation -------------------------------
        output = outcome.output
        resolution = resolve_citations(
            output.citations, pack, self._settings.max_citations
        )
        validation = self._citation_engine.validate(output.answer, resolution, pack)
        stages["total_ms"] = round(sum(stages.values()), 2)

        response = RAGResponse(
            query_id=query_id,
            question=question,
            retrieval_query=transformed.retrieval_query,
            transform_type=transformed.transform_type.value,
            mode="generated",
            answer=output.answer,
            citations=validation.final_citations,          # validated set, not raw
            confidence=output.confidence,
            insufficient_evidence=output.insufficient_evidence,
            degraded=retrieval_stats.degraded or resolution.dropped_count > 0,
            degraded_reason=(
                retrieval_stats.degraded_reason
                or (
                    f"{resolution.dropped_count} citation(s) failed resolution"
                    if resolution.dropped_count
                    else None
                )
            ),
            retrieval=retrieval_stats,
            stage_latencies_ms=stages,
            token_usage=outcome.usage,
            citation_validation=validation,                # Phase 12 addition
        )
    
    async def _attempt_answer(
        self,
        question: str,
        transformed: TransformedQuery,
        pack,                                    # type: ignore[no-untyped-def]
        retrieval_stats: RetrievalStats,
        history: Sequence[object],
        filter_,
        strict: bool = False,
    ) -> tuple:                                  # type: ignore[no-untyped-def]
        # ---- generation (Phase 11) -------------------------------------------
        start = time.perf_counter…trieval_stats.degraded or validation.dropped_count > 0,
            degraded_reason=(
                retrieval_stats.degraded_reason
                or (
                    f"{validation.dropped_count} citation(s) failed resolution"
                    if validation.dropped_count
                    else None
                )
            ),
            retrieval=retrieval_stats,
            stage_latencies_ms=stages,
            token_usage=outcome.usage,
            citation_validation=validation,
            grounding=GroundingMetrics(**grounding.__dict__) if grounding else None,
            grounding_policy=GroundingPolicyDecision(
                action=policy_decision.action.value,
                reason=policy_decision.reason,
                regeneration_attempts=regeneration_attempt,
                context_expanded=context_expanded,
            ) if policy_decision else None,
        