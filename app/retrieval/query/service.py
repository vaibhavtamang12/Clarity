"""QueryTransformService — routing + transforms + fail-safe orchestration.

Invariants:
- Retrieval ALWAYS gets a query: transformation never blocks or raises.
- Every LLM output is parsed, bounded, and sanitized before use (Rule 11).
- Everything is provenance: transform type, reasons, latency, token usage,
  degradation — ready for retrieval_logs and Phase 22 attribution.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import QueryTransformSettings
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.retrieval.query.decomposer import QueryDecomposer
from app.retrieval.query.domain import ConversationTurn, TransformedQuery, TransformType
from app.retrieval.query.expander import QueryExpander
from app.retrieval.query.rewriter import ConversationalRewriter
from app.retrieval.query.router import QueryRouter

logger = get_logger(__name__)


class QueryTransformService:
    def __init__(
        self,
        router: QueryRouter,
        rewriter: ConversationalRewriter,
        expander: QueryExpander,
        decomposer: QueryDecomposer,
        settings: QueryTransformSettings,
    ) -> None:
        self._router = router
        self._rewriter = rewriter
        self._expander = expander
        self._decomposer = decomposer
        self._settings = settings

    async def transform(
        self,
        query: str,
        history: Sequence[ConversationTurn] = (),
    ) -> TransformedQuery:
        user_turns = sum(1 for turn in history if turn.role == "user")
        decision = self._router.route(query, conversation_turns=user_turns)

        if not decision.needs_any:
            return TransformedQuery(
                original=query,
                retrieval_query=query,
                transform_type=TransformType.NONE,
                reasons=("router: no transformation needed",),
            )

        rewritten: str | None = None
        expansion_terms: tuple[str, ...] = ()
        sub_queries: tuple[str, ...] = ()
        total_latency = 0.0
        degraded_reasons: list[str] = []

        if decision.needs_conversational_rewrite:
            rewritten, latency = await self._rewriter.rewrite(query, history)
            total_latency += latency
            if rewritten is None:
                degraded_reasons.append("rewrite unavailable")

        if decision.needs_expansion:
            expansion_terms, latency = await self._expander.expand(query)
            total_latency += latency
            if not expansion_terms:
                degraded_reasons.append("expansion unavailable")

        if decision.needs_decomposition:
            sub_queries, latency = await self._decomposer.decompose(query)
            total_latency += latency
            if not sub_queries:
                degraded_reasons.append("decomposition unavailable")

        # ---- compose the retrieval query -------------------------------------
        retrieval_query = rewritten or query
        if expansion_terms:
            retrieval_query = f"{retrieval_query} {' '.join(expansion_terms)}".strip()

        applied = [
            name
            for name, active in (
                ("rewrite", rewritten is not None),
                ("expansion", bool(expansion_terms)),
                ("decomposition", bool(sub_queries)),
            )
            if active
        ]
        if len(applied) > 1:
            transform_type = TransformType.MULTI
        elif applied:
            transform_type = TransformType(applied[0])
        else:
            transform_type = TransformType.NONE

        result = TransformedQuery(
            original=query,
            retrieval_query=retrieval_query,
            transform_type=transform_type,
            rewritten=rewritten,
            expansion_terms=expansion_terms,
            sub_queries=sub_queries,
            reasons=decision.reasons,
            degraded=bool(degraded_reasons),
            degraded_reason="; ".join(degraded_reasons) if degraded_reasons else None,
            llm_latency_ms=round(total_latency, 2),
        )
        logger.info(
            "query_transform_completed",
            transform_type=result.transform_type.value,
            degraded=result.degraded,
            latency_ms=result.llm_latency_ms,
        )
        return result


def build_query_transform_service(
    provider: LLMProvider, settings: QueryTransformSettings
) -> QueryTransformService:
    return QueryTransformService(
        router=QueryRouter(settings),
        rewriter=ConversationalRewriter(provider, settings),
        expander=QueryExpander(provider, settings),
        decomposer=QueryDecomposer(provider, settings),
        settings=settings,
    )