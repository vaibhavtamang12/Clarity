"""API v1 router — rate limit applied to all authenticated routers."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.rate_limit import rate_limit_dependency
from app.api.v1.endpoints import (
    chat, chat_stream, conversations, documents, health, jobs, metrics, search,
)

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)  # health stays unauthenticated + unlimited

_protected = [Depends(rate_limit_dependency)]
api_v1_router.include_router(documents.router, dependencies=_protected)
api_v1_router.include_router(chat.router, dependencies=_protected)
api_v1_router.include_router(chat_stream.router, dependencies=_protected)
api_v1_router.include_router(search.router, dependencies=_protected)
api_v1_router.include_router(conversations.router, dependencies=_protected)
api_v1_router.include_router(jobs.router, dependencies=_protected)
api_v1_router.include_router(metrics.router, dependencies=_protected)