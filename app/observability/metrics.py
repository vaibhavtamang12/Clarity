"""Prometheus metrics registry (Phase 24, decision D-130).

All platform metrics are defined HERE — one module, one source of truth.
The observability package sits at the bottom of the layer stack: it imports
nothing above it, and upper layers call into it (D-131).

Cardinality discipline (D-136): labels come from a fixed vocabulary
(HTTP method, route template, branch name, outcome class). User IDs,
document IDs, and query text NEVER appear as labels.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# ------------------------------------------------------------------ HTTP layer
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled",
    ["method", "endpoint", "status_class"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# ------------------------------------------------------------- retrieval layer
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "retrieval_requests_total",
    "Total retrieval requests by retriever type",
    ["retriever", "degraded"],
    registry=REGISTRY,
)

RETRIEVAL_BRANCH_DURATION_SECONDS = Histogram(
    "retrieval_branch_duration_seconds",
    "Retrieval branch latency in seconds (dense/sparse/rerank)",
    ["branch"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    registry=REGISTRY,
)

RETRIEVED_CHUNKS = Histogram(
    "retrieval_returned_chunks",
    "Number of chunks returned per retrieval",
    buckets=(0, 1, 2, 5, 10, 20, 50),
    registry=REGISTRY,
)

# ------------------------------------------------------------- generation layer
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests by provider and outcome",
    ["provider", "outcome"],
    registry=REGISTRY,
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM generation latency in seconds",
    ["provider"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
    registry=REGISTRY,
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed by kind (prompt/completion)",
    ["provider", "kind"],
    registry=REGISTRY,
)

GENERATION_STAGE_DURATION_SECONDS = Histogram(
    "generation_stage_duration_seconds",
    "RAG pipeline stage latency in seconds",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

RAG_GROUNDING_SCORE = Histogram(
    "rag_grounding_score",
    "Distribution of grounding scores on generated answers",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# ------------------------------------------------------------------ cache layer
CACHE_EVENTS_TOTAL = Counter(
    "cache_events_total",
    "Cache events by cache and event type",
    ["cache", "event"],
    registry=REGISTRY,
)

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "rate_limit_rejections_total",
    "Total requests rejected by the rate limiter",
    registry=REGISTRY,
)

# ---------------------------------------------------------------- ingestion jobs
INGESTION_JOB_OUTCOMES_TOTAL = Counter(
    "ingestion_job_outcomes_total",
    "Ingestion job outcomes by type and result",
    ["job_type", "outcome"],
    registry=REGISTRY,
)

INGESTION_JOB_DURATION_SECONDS = Histogram(
    "ingestion_job_duration_seconds",
    "Ingestion job processing time in seconds",
    ["job_type"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
    registry=REGISTRY,
)

# ------------------------------------------------------------------ system health
DEPENDENCY_HEALTH = Gauge(
    "dependency_health",
    "Dependency health (1 = healthy, 0 = unavailable)",
    ["dependency"],
    registry=REGISTRY,
)

# Add to app/observability/metrics.py, with the other metric definitions:

SECURITY_EVENTS_TOTAL = Counter(
    "security_events_total",
    "Security-relevant events by type (injection detected, SSRF blocked, "
    "auth failure, malicious file rejected, ownership violation)",
    ["event_type"],
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Render all metrics in Prometheus exposition format."""
    return generate_latest(REGISTRY)