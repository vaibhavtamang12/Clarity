"""Request-scoped context propagation (foundation for Phase 24 tracing)."""

from __future__ import annotations

from contextvars import ContextVar, Token

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> Token[str | None]:
    return request_id_var.set(request_id)


def get_request_id() -> str | None:
    return request_id_var.get()