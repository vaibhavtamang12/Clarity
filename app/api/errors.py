"""Exception → HTTP error envelope mapping (ARCHITECTURE.md section 6).

All errors — domain, validation, FastAPI HTTP, and unexpected — are normalized
into one machine-readable shape. Unexpected errors never leak internals:
the client gets a generic message; the stack trace stays in the logs (Rule 10).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

_STATUS_TO_CODE: dict[int, ErrorCode] = {
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    429: ErrorCode.RATE_LIMITED,
}


def _envelope(
    code: str,
    message: str,
    request_id: str | None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        error["details"] = details
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "domain_error",
            code=exc.code.value,
            status=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code.value, exc.message, request_id, exc.details or None),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # Echo locations/types/messages only — never raw input values (data-leak hygiene).
        field_errors = [
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "type": err.get("type"),
                "msg": err.get("msg"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope(
                ErrorCode.VALIDATION_ERROR.value,
                "Request validation failed",
                request_id,
                {"fields": field_errors},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.HTTP_ERROR)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code.value, str(exc.detail), request_id),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled_error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=_envelope(ErrorCode.INTERNAL.value, "Internal server error", request_id),
        )