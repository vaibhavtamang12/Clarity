"""Failure taxonomy for ingestion jobs (Phase 19).

Two classes drive everything:
- PERMANENT: retrying cannot succeed (malformed content, unsupported type,
  invalid configuration). Fail fast, mark the document FAILED, send to DLQ.
- TRANSIENT: infrastructure-style failures (embedding service, vector store,
  LLM, network, timeouts). Retry with exponential backoff + jitter.

Unknown errors default to TRANSIENT: it is cheaper to waste a retry than to
permanently fail a job that a restart would have fixed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from app.core.exceptions import (
    ConfigurationError,
    EmbeddingUnavailableError,
    ExternalServiceError,
    IngestionValidationError,
    LLMUnavailableError,
    ParseError,
    RetrievalUnavailableError,
    UnsupportedDocumentTypeError,
    VectorStoreUnavailableError,
)


class FailureClass(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


PERMANENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ParseError,
    UnsupportedDocumentTypeError,
    IngestionValidationError,
    ConfigurationError,
    ValueError,
)

TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    EmbeddingUnavailableError,
    VectorStoreUnavailableError,
    LLMUnavailableError,
    RetrievalUnavailableError,
    ExternalServiceError,
    TimeoutError,
    ConnectionError,
    OSError,
)

MAX_BACKOFF_SECONDS = 300.0


def classify_error(exc: BaseException) -> FailureClass:
    if isinstance(exc, PERMANENT_EXCEPTIONS):
        return FailureClass.PERMANENT
    if isinstance(exc, TRANSIENT_EXCEPTIONS):
        return FailureClass.TRANSIENT
    return FailureClass.TRANSIENT  # unknown → retry a bounded number of times


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: float


def decide_retry(
    attempt_count: int,
    max_attempts: int,
    base_delay_seconds: float,
) -> RetryDecision:
    """Exponential backoff with full jitter, capped at MAX_BACKOFF_SECONDS.

    attempt_count is the number of attempts ALREADY made; the next attempt
    would be number attempt_count + 1.
    """
    next_attempt = attempt_count + 1
    if next_attempt >= max_attempts:
        return RetryDecision(should_retry=False, delay_seconds=0.0)
    delay = min(
        MAX_BACKOFF_SECONDS,
        base_delay_seconds * (2 ** attempt_count) + random.uniform(0.0, 1.0),
    )
    return RetryDecision(should_retry=True, delay_seconds=round(delay, 2))