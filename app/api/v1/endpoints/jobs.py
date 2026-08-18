# app/api/v1/endpoints/jobs.py
"""Job status endpoint (polling for async ingestion)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db
from app.core.exceptions import NotFoundError
from app.repositories.document import DocumentRepository
from app.repositories.job import IngestionJobRepository
from app.schemas.jobs import JobStatusResponse
from app.api.deps import get_platform

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: uuid.UUID,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> JobStatusResponse:
    job = await IngestionJobRepository(session).get_by_id(job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    # Ownership via the parent document — 404 for foreign jobs (D-085).
    doc = await DocumentRepository(session).get_by_id(job.document_id)
    if doc is None or doc.owner_id != user.id:
        raise NotFoundError(f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.id, document_id=job.document_id,
        job_type=job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        attempt_count=job.attempt_count, max_attempts=job.max_attempts,
        error_type=job.error_type, error_message=job.error_message,
        progress = dict(job.progress or {})
        platform = request.app.state.platform if hasattr(request, "app") else None,
        created_at=job.created_at, started_at=job.started_at, completed_at=job.completed_at,
    )