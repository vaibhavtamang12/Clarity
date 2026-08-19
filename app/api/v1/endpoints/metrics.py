"""Metrics endpoints (Phase 24).

- GET /metrics          — Prometheus exposition format, UNAUTHENTICATED
                          (scrape target; network-level protection in prod, D-135)
- GET /metrics/summary  — authenticated JSON operational summary
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response as RawResponse
from sqlalchemy import func, select

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.models.conversation import Conversation
from app.models.document import Document, DocumentVersion
from app.models.job import IngestionJob
from app.observability.metrics import render_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("", include_in_schema=False)
async def prometheus_metrics() -> RawResponse:
    """Prometheus scrape endpoint (D-135)."""
    return RawResponse(content=render_metrics(), media_type=PROMETHEUS_CONTENT_TYPE)


@router.get("/summary")
async def metrics_summary(
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Authenticated operational summary (JSON)."""
    platform = get_platform(request)
    counters = getattr(request.app.state, "metrics", {})

    job_rows = (
        await session.execute(
            select(IngestionJob.status, func.count()).group_by(IngestionJob.status)
        )
    ).all()
    jobs_by_status = {
        (status.value if hasattr(status, "value") else str(status)): count
        for status, count in job_rows
    }

    async def _count(model) -> int:  # type: ignore[no-untyped-def]
        return int((await session.execute(select(func.count()).select_from(model))).scalar_one())

    return {
        "service": platform.settings.app.name,
        "version": platform.settings.app.version,
        "requests": dict(counters),
        "jobs": jobs_by_status,
        "documents": await _count(Document),
        "document_versions": await _count(DocumentVersion),
        "conversations": await _count(Conversation),
    }