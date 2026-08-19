"""Structured logging policy (Phase 24, decision D-134).

The rules, enforced by code rather than convention:
1. Document content, chunk text, and generated answers NEVER enter logs.
2. Question text is DEBUG-level only; INFO logs carry IDs and metadata.
3. Errors are logged by exception CLASS NAME, not message bodies that may
   contain user content.
4. Every request-scoped log carries request_id; query-scoped logs add
   query_id and conversation_id.

sanitize_fields() is the chokepoint: any structured log event that might
touch request-derived data passes through it.
"""

from __future__ import annotations

from typing import Any

import structlog

# Keys whose values must never be logged.
SENSITIVE_KEYS = frozenset(
    {
        "content",
        "answer",
        "question",
        "chunk_content",
        "document_content",
        "passage",
        "password",
        "api_key",
        "token",
        "secret",
        "authorization",
    }
)

# Keys that are DEBUG-only (allowed, but stripped at INFO and above).
DEBUG_ONLY_KEYS = frozenset({"question_text", "rewritten_query_text"})


def sanitize_fields(fields: dict[str, Any], level: str = "INFO") -> dict[str, Any]:
    """Strip sensitive keys; drop DEBUG-only keys unless logging at DEBUG."""
    cleaned: dict[str, Any] = {}
    level_upper = level.upper()
    for key, value in fields.items():
        if key in SENSITIVE_KEYS:
            cleaned[key] = "[redacted]"
            continue
        if key in DEBUG_ONLY_KEYS and level_upper != "DEBUG":
            continue
        cleaned[key] = value
    return cleaned


def bind_query_context(
    query_id: str,
    conversation_id: str | None = None,
    request_id: str | None = None,
) -> None:
    """Bind query-scoped identifiers into the structlog context."""
    context: dict[str, Any] = {"query_id": query_id}
    if conversation_id is not None:
        context["conversation_id"] = conversation_id
    if request_id is not None:
        context["request_id"] = request_id
    structlog.contextvars.bind_contextvars(**context)


def clear_query_context() -> None:
    structlog.contextvars.unbind_contextvars(
        "query_id", "conversation_id", "request_id"
    )