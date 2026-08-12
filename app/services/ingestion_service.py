"""Submission-side orchestration: validation → storage → document/job rows.

This is what the API endpoints (Phase 16) call. Submission is cheap and
synchronous; the heavy work happens when a worker runs the job.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import IngestionValidationError
from app.core.logging import get_logger
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.idempotency import compute_idempotency_key
from app.ingestion.domain import sha256_hex
from app.ingestion.storage import FileStore
from app.ingestion.url_fetcher import fetch_url
from app.ingestion.validation import detect_source_type, validate_upload
from app.models.enums import DocumentSourceType, DocumentStatus
from app.models.job import IngestionJob
from app.repositories.document import DocumentRepository
from app.repositories.job import IngestionJobRepository

logger = get_logger(__name__)

_CONTENT_TYPE_TO_SOURCE: dict[str, DocumentSourceType] = {
    "text/html": DocumentSourceType.HTML,
    "text/plain": DocumentSourceType.TXT,
    "text/markdown": DocumentSourceType.MARKDOWN,
    "application/pdf": DocumentSourceType.PDF,
}


class IngestionService:
    def __init__(
        self,
        file_store: FileStore,
        chunking_registry: ChunkingRegistry,
        settings: Settings,
    ) -> None:
        self.file_store = file_store
        self.chunking_registry = chunking_registry
        self.settings = settings

    async def submit_file(
        self,
        *,
        session: AsyncSession,
        owner_id: uuid.UUID,
        filename: str,
        content: bytes,
    ) -> IngestionJob:
        source_type = validate_upload(
            filename, len(content), content, self.settings.ingestion
        )
        return await self._submit(
            session=session,
            owner_id=owner_id,
            title=filename,
            source_uri=filename,
            source_type=source_type,
            content=content,
        )

    async def submit_url(
        self, *, session: AsyncSession, owner_id: uuid.UUID, url: str
    ) -> IngestionJob:
        # Fetch at submission time: retries replay stored bytes, never re-fetch
        # (deterministic, and no repeated SSRF exposure — decision D-025).
        fetched = await fetch_url(url, self.settings.ingestion)
        source_type = self._source_type_for_fetch(fetched.content_type, url)
        title = Path(url.rstrip("/")).name or url
        return await self._submit(
            session=session,
            owner_id=owner_id,
            title=title,
            source_uri=fetched.final_url,
            source_type=source_type,
            content=fetched.content,
        )

    async def _submit(
        self,
        *,
        session: AsyncSession,
        owner_id: uuid.UUID,
        title: str,
        source_uri: str,
        source_type: DocumentSourceType,
        content: bytes,
    ) -> IngestionJob:
        documents = DocumentRepository(session)
        jobs = IngestionJobRepository(session)

        content_hash = sha256_hex(content)
        strategy = self.chunking_registry.strategies[self.chunking_registry.default]
        idempotency_key = compute_idempotency_key(
            content_hash=content_hash,
            parser=source_type.value,
            chunking_strategy=self.chunking_registry.default,
            chunking_config=strategy.model_dump(mode="json"),
            embedding_model=self.settings.embedding.default_model,
        )

        existing = await jobs.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            logger.info("ingestion_submit_deduplicated", job_id=str(existing.id))
            return existing

        document = await documents.create(
            owner_id=owner_id,
            source_type=source_type.value,
            title=title,
            source_uri=source_uri,
        )
        document.status = DocumentStatus.PENDING
        document.content_hash = content_hash
        await session.flush()

        self.file_store.save(str(document.id), content)

        job = await jobs.create(document.id, idempotency_key=idempotency_key)
        await jobs.enqueue(job)
        await session.flush()
        logger.info(
            "ingestion_submitted",
            job_id=str(job.id),
            document_id=str(document.id),
            source_type=source_type.value,
        )
        return job

    @staticmethod
    def _source_type_for_fetch(content_type: str, url: str) -> DocumentSourceType:
        base_type = content_type.split(";")[0].strip().lower()
        if base_type in _CONTENT_TYPE_TO_SOURCE:
            return _CONTENT_TYPE_TO_SOURCE[base_type]
        suffix = Path(url.rstrip("/")).suffix.lower()
        if suffix:
            try:
                return detect_source_type(f"file{suffix}")
            except IngestionValidationError:
                pass
        return DocumentSourceType.HTML  # default assumption for web content