"""Request-scoped middleware."""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.context import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=request.method, path=request.url.path,
        )
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
            structlog.contextvars.clear_contextvars()
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Aggregate request counters (Phase 16 minimum; Prometheus in Phase 24)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        counters = getattr(request.app.state, "metrics", None)
        if counters is not None:
            counters["requests_total"] = counters.get("requests_total", 0) + 1
            if response.status_code >= 500:
                counters["server_errors_total"] = counters.get("server_errors_total", 0) + 1
            elif response.status_code >= 400:
                counters["client_errors_total"] = counters.get("client_errors_total", 0) + 1
        return response