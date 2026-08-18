# app/api/v1/endpoints/metrics.py
"""Operational metrics endpoint (JSON now; Prometheus exposition in Phase 24)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.models.conversation import Conversation
from app.models.document import Document, DocumentVersion
from app.models.job import IngestionJob

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def metrics(
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
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