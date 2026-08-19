"""Executes ingestion jobs: claim → lock → run → complete / retry / fail.

Phase 19 execution model — two entry paths share one execution core:
- run_next():            DB polling sweep (authoritative; fallback + tests)
- run_specific(job_id):  stream-triggered execution (locked claim by id)

Failure handling is taxonomy-driven (app/workers/failures.py):
- PERMANENT errors fail fast — no wasted retries — and go to the DLQ
- TRANSIENT errors retry with exponential backoff + jitter
- exhausted jobs are FAILED in PostgreSQL and mirrored to the DLQ

The document lock wraps processing: if another worker holds the document,
the claim is released back to QUEUED and the sweep retries later. No
starvation, no concurrent processing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
import time
from sqlalchemy import select

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import FileStore
from app.models.enums import DocumentStatus, JobStatus, JobType
from app.models.job import IngestionJob
from app.repositories.database import Database
from app.repositories.document import DocumentRepository, DocumentVersionRepository
from app.repositories.job import IngestionJobRepository
from app.services.indexing_service import IndexingService
from app.services.job_state import JobStateStore
from app.workers.failures import FailureClass, classify_error, decide_retry
from app.workers.protocols import DeadLetterSink, DocumentLockManager
from app.observability.instrumentation import instrument_job_outcome


logger = get_logger(__name__)

_MAX_STORED_ERROR_CHARS = 2000


class IngestionJobRunner:
    def __init__(
        self,
        database: Database,
        pipeline: IngestionPipeline,
        file_store: FileStore,
        settings: Settings,
        indexer: IndexingService | None = None,
        job_state_store: JobStateStore | None = None,
        locks: DocumentLockManager | None = None,          # Phase 19
        dlq: DeadLetterSink | None = None,                 # Phase 19
    ) -> None:
        self.database = database
        self.pipeline = pipeline
        self.file_store = file_store
        self.settings = settings
        self.indexer = indexer
        self.job_state_store = job_state_store
        self.locks = locks
        self.dlq = dlq

    # ------------------------------------------------------------ entry paths
    async def run_next(self) -> bool:
        """Claim and process the next due job from PostgreSQL (sweep path)."""
        async with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            job = await jobs.claim_next()
            if job is None:
                return False
            await self._execute_in_session(session, jobs, job)
            return True

    async def run_specific(self, job_id: uuid.UUID) -> bool:
        """Process one job by id (stream path). Idempotent: jobs that are
        already claimed/completed/cancelled are skipped, which makes
        at-least-once redelivery safe."""
        async with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            result = await session.execute(
                select(IngestionJob)
                .where(IngestionJob.id == job_id, IngestionJob.status == JobStatus.QUEUED)
                .with_for_update(skip_locked=True)
            )
            job = result.scalar_one_or_none()
            if job is None:
                return False  # not claimable → duplicate delivery or terminal state
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await session.flush()
            await self._execute_in_session(session, jobs, job)
            return True

    # ----------------------------------------------------------- execution core
    async def _execute_in_session(
        self, session, jobs: IngestionJobRepository, job: IngestionJob  # type: ignore[no-untyped-def]
    ) -> None:
        lock_token: str | None = None
        document_id_str = str(job.document_id)
        job_type = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)

        if self.locks is not None:
            lock_token = await self.locks.try_acquire(document_id_str)
            if lock_token is None:
                job.status = JobStatus.QUEUED
                job.started_at = None
                await session.commit()
                instrument_job_outcome(job_type, "deferred")
                logger.info("job_deferred_document_busy", job_id=str(job.id))
                return

        started = time.perf_counter()
        try:
            await self._process(session, job)
            await jobs.mark_completed(job)
            await session.commit()
            instrument_job_outcome(job_type, "completed", time.perf_counter() - started)
            logger.info("job_completed", job_id=str(job.id))
        except Exception as exc:  # noqa: BLE001 — failures are data, not crashes
            await session.rollback()
            await self._record_failure(job.id, job.document_id, exc, job_type, started)
        finally:
            if self.locks is not None and lock_token is not None:
                await self.locks.release(document_id_str, lock_token)

    async def _process(self, session, job: IngestionJob) -> None:  # type: ignore[no-untyped-def]
        documents = DocumentRepository(session)
        jobs = IngestionJobRepository(session)

        document = await documents.get_by_id(job.document_id)
        if document is None:
            raise ValueError(f"Document {job.document_id} missing for job {job.id}")

        job_type = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)

        # ---- REINDEX: re-embed + re-upsert the active version ---------------
        if job_type == JobType.REINDEX.value:
            if self.indexer is None:
                raise ValueError("Indexer not configured — cannot reindex")
            await jobs.update_progress(job, "reindexing", 40)
            active = await DocumentVersionRepository(session).get_active_for_document(document.id)
            if active is None:
                raise ValueError(f"No active version to reindex for document {document.id}")
            await self.indexer.index_version(session, active.id)
            await jobs.update_progress(job, "completed", 100)
            return

        # ---- INGEST ------------------------------------------------------------
        await jobs.update_progress(job, "parsing", 10)
        content = self.file_store.load(str(document.id))
        document.status = DocumentStatus.PROCESSING
        await session.flush()

        result = await self.pipeline.ingest(
            session=session, document=document, content=content
        )
        job.version_id = result.version_id

        if self.indexer is not None and result.version_id is not None and result.changed:
            await jobs.update_progress(job, "embedding_indexing", 60)
            await self.indexer.index_version(session, result.version_id)

        await jobs.update_progress(job, "completed", 100)

    # ----------------------------------------------------------- failure paths
    async def _record_failure(
        self,
        job_id: uuid.UUID,
        document_id: uuid.UUID,
        exc: Exception,
        job_type: str = "ingest",
        started: float | None = None,
    ) -> None:
        from app.workers.failures import FailureClass, classify_error, decide_retry

        failure_class = classify_error(exc)
        error_type = type(exc).__name__
        error_message = str(exc)[:_MAX_STORED_ERROR_CHARS]
        duration = (time.perf_counter() - started) if started is not None else None

        async with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            job = await jobs.get_by_id(job_id)
            if job is None:
                return

            if failure_class == FailureClass.PERMANENT:
                job.attempt_count += 1
                job.status = JobStatus.FAILED
                job.error_type = error_type
                job.error_message = error_message
                job.completed_at = datetime.now(timezone.utc)
                await documents.update_status(document_id, DocumentStatus.FAILED)
                await session.commit()
                instrument_job_outcome(job_type, "failed", duration)
                logger.warning("job_failed_permanent", job_id=str(job_id), error_type=error_type)
                await self._send_dlq(job_id, error_type, error_message, "permanent_error")
                return

            decision = decide_retry(
                attempt_count=job.attempt_count,
                max_attempts=job.max_attempts,
                base_delay_seconds=self.settings.ingestion.retry_base_delay_seconds,
            )
            if decision.should_retry:
                await jobs.mark_failed(
                    job,
                    error_type=error_type,
                    error_message=error_message,
                    retry_delay_seconds=decision.delay_seconds,
                )
                await session.commit()
                logger.warning(
                    "job_failed_will_retry",
                    job_id=str(job_id),
                    error_type=error_type,
                    attempt=job.attempt_count,
                    retry_in_seconds=decision.delay_seconds,
                )
                return

            await jobs.mark_failed(job, error_type=error_type, error_message=error_message)
            await documents.update_status(document_id, DocumentStatus.FAILED)
            await session.commit()
            instrument_job_outcome(job_type, "retries_exhausted", duration)
            logger.warning(
                "job_failed_retries_exhausted",
                job_id=str(job_id),
                error_type=error_type,
                attempts=job.attempt_count,
            )
            await self._send_dlq(job_id, error_type, error_message, "retries_exhausted")

    async def _send_dlq(
        self, job_id: uuid.UUID, error_type: str, error_message: str, reason: str
    ) -> None:
        if self.dlq is None:
            return
        try:
            await self.dlq.send_dead_letter(job_id, error_type, error_message, reason)
        except Exception as exc:  # noqa: BLE001 — DLQ failure must not mask job state
            logger.warning("dlq_write_failed", error=type(exc).__name__)