"""IngestionJob repository — queue semantics live here.

claim_next() uses SELECT ... FOR UPDATE SKIP LOCKED: multiple workers can
claim concurrently without blocking each other or double-claiming.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select

from app.models.enums import JobStatus
from app.models.job import IngestionJob
from app.repositories.base import BaseRepository


class IngestionJobRepository(BaseRepository[IngestionJob]):
    model = IngestionJob

    async def create(
        self,
        document_id: uuid.UUID,
        idempotency_key: str,
        job_type: str = "ingest",
        priority: int = 0,
        max_attempts: int = 3,
    ) -> IngestionJob:
        job = IngestionJob(
            document_id=document_id,
            idempotency_key=idempotency_key,
            job_type=job_type,  # type: ignore[arg-type]
            priority=priority,
            max_attempts=max_attempts,
        )
        return await self.add(job)

    async def get_by_idempotency_key(self, key: str) -> IngestionJob | None:
        result = await self.session.execute(
            select(IngestionJob).where(IngestionJob.idempotency_key == key)
        )
        return result.scalar_one_or_none()

    async def enqueue(self, job: IngestionJob) -> None:
        job.status = JobStatus.QUEUED
        await self.session.flush()

    async def claim_next(self) -> IngestionJob | None:
        """Claim the highest-priority due job without blocking other workers."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(IngestionJob)
            .where(IngestionJob.status == JobStatus.QUEUED)
            .where(
                or_(IngestionJob.next_retry_at.is_(None), IngestionJob.next_retry_at <= now)
            )
            .order_by(IngestionJob.priority.desc(), IngestionJob.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is not None:
            job.status = JobStatus.PROCESSING
            job.started_at = now
            await self.session.flush()
        return job

    async def mark_completed(self, job: IngestionJob) -> None:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def mark_failed(
        self,
        job: IngestionJob,
        error_type: str,
        error_message: str,
        retry_delay_seconds: float | None = None,
    ) -> None:
        """Fail a job. If retries remain, schedule the next attempt with backoff."""
        job.attempt_count += 1
        job.error_type = error_type
        job.error_message = error_message
        if job.attempt_count < job.max_attempts and retry_delay_seconds is not None:
            job.status = JobStatus.QUEUED
            job.next_retry_at = datetime.now(timezone.utc) + timedelta(
                seconds=retry_delay_seconds
            )
        else:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def update_progress(self, job: IngestionJob, stage: str, percent: int) -> None:
        job.progress = {"stage": stage, "percent": percent}
        await self.session.flush()

    async def list_stuck(self, processing_before: datetime) -> list[IngestionJob]:
        """Jobs crashed mid-processing — reclaimed by the reaper (ARCHITECTURE §9)."""
        result = await self.session.execute(
            select(IngestionJob).where(
                and_(
                    IngestionJob.status == JobStatus.PROCESSING,
                    IngestionJob.started_at < processing_before,
                )
            )
        )
        return list(result.scalars().all())