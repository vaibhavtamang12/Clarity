"""Executes ingestion jobs: claim → run pipeline → complete / retry / fail.

The Phase 19 worker process will simply loop over run_next() and listen to
the Redis stream; the execution semantics live here and are tested now.

Retry policy: exponential backoff with jitter, max_attempts from settings.
After a failure the transaction is rolled back (job returns to QUEUED) and
a fresh transaction records the attempt + next_retry_at — no partial state.
"""

from __future__ import annotations

import random

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import FileStore
from app.models.enums import DocumentStatus
from app.repositories.database import Database
from app.repositories.document import DocumentRepository
from app.repositories.job import IngestionJobRepository

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
    ) -> None:
        self.database = database
        self.pipeline = pipeline
        self.file_store = file_store
        self.settings = settings
        self.indexer = indexer

    async def run_next(self) -> bool:
        """Claim and process one job. Returns False when the queue is empty."""
        async with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            job = await jobs.claim_next()
            if job is None:
                return False
            document_id = job.document_id
            job_id = job.id
            try:
                await self._process(session, job)
                await jobs.mark_completed(job)
                await session.commit()
                logger.info("job_completed", job_id=str(job_id))
                return True
            except Exception as exc:  # noqa: BLE001 — job failures are data, not crashes
                await session.rollback()
                await self._record_failure(job_id, document_id, exc)
                return True

     async def _process(self, session, job) -> None:  # type: ignore[no-untyped-def]
        documents = DocumentRepository(session)
        jobs = IngestionJobRepository(session)

        document = await documents.get_by_id(job.document_id)
        if document is None:
            raise ValueError(f"Document {job.document_id} missing for job {job.id}")

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

    async def _record_failure(self, job_id, document_id, exc: Exception) -> None:  # type: ignore[no-untyped-def]
        attempt_delay = self.settings.ingestion.retry_base_delay_seconds
        async with self.database.session() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            job = await jobs.get_by_id(job_id)
            if job is None:
                return
            delay = attempt_delay * (2 ** job.attempt_count) + random.uniform(0.0, 1.0)
            await jobs.mark_failed(
                job,
                error_type=type(exc).__name__,
                error_message=str(exc)[:_MAX_STORED_ERROR_CHARS],
                retry_delay_seconds=delay,
            )
            if job.status.value == "failed":  # retries exhausted
                await documents.update_status(document_id, DocumentStatus.FAILED)
            await session.commit()
        logger.warning(
            "job_failed",
            job_id=str(job_id),
            error_type=type(exc).__name__,
            attempt=job.attempt_count,
        )