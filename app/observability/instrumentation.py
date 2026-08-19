"""Instrumentation chokepoints (Phase 24, decision D-133).

Five call sites cover the whole platform:
1. HTTP middleware        → instrument_http
2. RAG response           → instrument_rag_response (retrieval + stages + tokens + grounding)
3. Cache                  → instrument_cache_event
4. Ingestion job outcomes → instrument_job_outcome
5. Health probes          → set_dependency_health

Upper layers import this module (allowed by the layer contract); this
module imports only the metrics registry.
"""

from __future__ import annotations

from app.generation.domain import RAGResponse
from app.observability import metrics
from app.retrieval.base import RetrievalMetadata

_RETRIEVAL_BRANCHES = ("dense", "sparse", "rerank")


def instrument_http(
    method: str, route_template: str, status_code: int, duration_seconds: float
) -> None:
    status_class = f"{status_code // 100}xx"
    metrics.HTTP_REQUESTS_TOTAL.labels(
        method=method, endpoint=route_template, status_class=status_class
    ).inc()
    metrics.HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method, endpoint=route_template
    ).observe(duration_seconds)


def instrument_retrieval_metadata(metadata: RetrievalMetadata) -> None:
    metrics.RETRIEVAL_REQUESTS_TOTAL.labels(
        retriever=metadata.retriever, degraded=str(metadata.degraded).lower()
    ).inc()
    for branch in _RETRIEVAL_BRANCHES:
        latency_ms = metadata.branch_latencies_ms.get(branch)
        if latency_ms is not None:
            metrics.RETRIEVAL_BRANCH_DURATION_SECONDS.labels(branch=branch).observe(
                latency_ms / 1000.0
            )


def instrument_rag_response(response: RAGResponse, llm_provider: str = "llm") -> None:
    """Instrument a completed RAG response across all metric families."""
    # ---- retrieval ------------------------------------------------------------
    instrument_retrieval_metadata(response.retrieval)
    metrics.RETRIEVED_CHUNKS.observe(response.retrieval.num_items)

    # ---- pipeline stage latencies ----------------------------------------------
    for stage, latency_ms in response.stage_latencies_ms.items():
        metrics.GENERATION_STAGE_DURATION_SECONDS.labels(stage=stage).observe(
            latency_ms / 1000.0
        )

    # ---- LLM latency + tokens ----------------------------------------------------
    generation_ms = response.stage_latencies_ms.get("generation_ms")
    outcome = "fallback" if response.mode == "retrieval_only" else "generated"
    metrics.LLM_REQUESTS_TOTAL.labels(provider=llm_provider, outcome=outcome).inc()
    if generation_ms is not None:
        metrics.LLM_REQUEST_DURATION_SECONDS.labels(provider=llm_provider).observe(
            generation_ms / 1000.0
        )
    if response.token_usage is not None:
        metrics.LLM_TOKENS_TOTAL.labels(provider=llm_provider, kind="prompt").inc(
            response.token_usage.prompt_tokens
        )
        metrics.LLM_TOKENS_TOTAL.labels(provider=llm_provider, kind="completion").inc(
            response.token_usage.completion_tokens
        )

    # ---- grounding ----------------------------------------------------------------
    if response.grounding is not None:
        metrics.RAG_GROUNDING_SCORE.observe(response.grounding.score)


def instrument_cache_event(cache: str, event: str) -> None:
    """cache: response | embedding — event: hit | miss | put | error"""
    metrics.CACHE_EVENTS_TOTAL.labels(cache=cache, event=event).inc()


def instrument_job_outcome(job_type: str, outcome: str, duration_seconds: float | None = None) -> None:
    """outcome: completed | failed | deferred | retries_exhausted"""
    metrics.INGESTION_JOB_OUTCOMES_TOTAL.labels(job_type=job_type, outcome=outcome).inc()
    if duration_seconds is not None:
        metrics.INGESTION_JOB_DURATION_SECONDS.labels(job_type=job_type).observe(
            duration_seconds
        )


def set_dependency_health(dependency: str, healthy: bool) -> None:
    metrics.DEPENDENCY_HEALTH.labels(dependency=dependency).set(1 if healthy else 0)