"""VersionService — user-facing document versioning orchestration (Phase 15).

Composes primitives built in Phases 3/4/7 into the complete lifecycle the
spec demands: detect changes → create new version → reprocess → update
indexes → preserve history — plus inspection, diffing, and rollback.

Design:
- submit_update is DOUBLE-idempotent (D-082): identical content is a no-op
  both here (content_hash vs latest version) and in the pipeline (Phase 4).
- rollback reactivates a historical version with ZERO reprocessing:
  DB status handshake + vector payload flip + index-state bump.
- Every mutation that changes what retrieval can see bumps the index state
  counter, so Phase 18's cache keys can never serve stale answers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ConflictError, DocumentNotFoundError, IngestionValidationError
from app.core.logging import get_logger
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.domain import sha256_hex
from app.ingestion.idempotency import compute_idempotency_key
from app.ingestion.storage import FileStore
from app.ingestion.validation import detect_source_type, validate_upload
from app.ingestion.version_diff import VersionDiff, diff_chunks
from app.models.document import DocumentVersion
from app.models.enums import DocumentStatus, VersionStatus
from app.repositories.document import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
)
from app.repositories.job import IngestionJobRepository
from app.services.index_state import IndexStateNotifier
from app.services.indexing_service import IndexingService

logger = get_logger(__name__)


@dataclass(frozen=True)
class VersionSubmitResult:
    changed: bool
    job_id: uuid.UUID | None
    existing_version_number: int | None   # set when content is unchanged
    expected_version_number: int | None   # set when a new version is queued


@dataclass(frozen=True)
class VersionInfo:
    version_number: int
    status: str
    content_hash: str
    chunk_count: int | None
    token_count: int | None
    page_count: int | None
    embedding_model: str | None
    parser: str | None
    created_at: datetime


class VersioningService:
    def __init__(
        self,
        file_store: FileStore,
        chunking_registry: ChunkingRegistry,
        settings: Settings,
        indexer: IndexingService,
        index_state: IndexStateNotifier,
    ) -> None:
        self._file_store = file_store
        self._chunking_registry = chunking_registry
        self._settings = settings
        self._indexer = indexer
        self._index_state = index_state

    # ------------------------------------------------------------ submit update
    async def submit_update(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
        content: bytes,
        filename: str,
    ) -> VersionSubmitResult:
        documents = DocumentRepository(session)
        versions = DocumentVersionRepository(session)
        jobs = IngestionJobRepository(session)

        document = await documents.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        # ---- validate: re-upload must match the document's source type ------
        detected_type = validate_upload(
            filename, len(content), content, self._settings.ingestion
        )
        current_type = (
            document.source_type.value
            if hasattr(document.source_type, "value")
            else str(document.source_type)
        )
        if detected_type.value != current_type:
            raise IngestionValidationError(
                f"Re-uploaded file type '{detected_type.value}' does not match "
                f"document type '{current_type}'"
            )

        # ---- change detection (first idempotency gate, D-082) ----------------
        content_hash = sha256_hex(content)
        latest = await versions.get_latest_for_document(document_id)
        if latest is not None and latest.content_hash == content_hash:
            logger.info(
                "version_submit_unchanged",
                document_id=str(document_id),
                version=latest.version_number,
            )
            return VersionSubmitResult(
                changed=False,
                job_id=None,
                existing_version_number=latest.version_number,
                expected_version_number=None,
            )

        # ---- deduplicate against an already-queued identical job -------------
        strategy = self._chunking_registry.strategies[self._chunking_registry.default]
        idempotency_key = compute_idempotency_key(
            content_hash=content_hash,
            parser=current_type,
            chunking_strategy=self._chunking_registry.default,
            chunking_config=strategy.model_dump(mode="json"),
            embedding_model=self._settings.embedding.default_model,
        )
        existing_job = await jobs.get_by_idempotency_key(idempotency_key)
        if existing_job is not None:
            return VersionSubmitResult(
                changed=True,
                job_id=existing_job.id,
                existing_version_number=None,
                expected_version_number=(latest.version_number + 1) if latest else 1,
            )

        # ---- store bytes + queue the ingestion job ----------------------------
        self._file_store.save(str(document_id), content)
        job = await jobs.create(document_id, idempotency_key=idempotency_key)
        await jobs.enqueue(job)
        await session.flush()

        expected = (latest.version_number + 1) if latest else 1
        logger.info(
            "version_update_submitted",
            document_id=str(document_id),
            job_id=str(job.id),
            expected_version=expected,
        )
        return VersionSubmitResult(
            changed=True,
            job_id=job.id,
            existing_version_number=None,
            expected_version_number=expected,
        )

    # ------------------------------------------------------------------ inspect
    async def list_versions(
        self, *, session: AsyncSession, document_id: uuid.UUID
    ) -> list[VersionInfo]:
        versions = DocumentVersionRepository(session)
        all_versions = await versions.list_for_document(document_id)
        return [
            VersionInfo(
                version_number=v.version_number,
                status=v.status.value if hasattr(v.status, "value") else str(v.status),
                content_hash=v.content_hash,
                chunk_count=v.chunk_count,
                token_count=v.token_count,
                page_count=v.page_count,
                embedding_model=v.embedding_model,
                parser=v.parser,
                created_at=v.created_at,
            )
            for v in all_versions
        ]

    async def get_version(
        self, *, session: AsyncSession, document_id: uuid.UUID, version_number: int
    ) -> DocumentVersion:
        versions = DocumentVersionRepository(session)
        version = await versions.get_by_number(document_id, version_number)
        if version is None:
            raise DocumentNotFoundError(
                f"Version {version_number} of document {document_id} not found"
            )
        return version

    # ------------------------------------------------------------------- compare
    async def compare_versions(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
        from_version: int,
        to_version: int,
    ) -> VersionDiff:
        """Chunk-level diff between two versions (multiset semantics)."""
        old = await self.get_version(
            session=session, document_id=document_id, version_number=from_version
        )
        new = await self.get_version(
            session=session, document_id=document_id, version_number=to_version
        )
        chunks = DocumentChunkRepository(session)
        old_hashes = await chunks.list_content_hashes_for_version(old.id)
        new_hashes = await chunks.list_content_hashes_for_version(new.id)
        diff = diff_chunks(old_hashes, new_hashes)
        logger.info(
            "versions_compared",
            document_id=str(document_id),
            from_version=from_version,
            to_version=to_version,
            added=diff.added,
            removed=diff.removed,
            unchanged=diff.unchanged,
        )
        return diff

    # ------------------------------------------------------------------ rollback
    async def rollback(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
        version_number: int,
    ) -> DocumentVersion:
        """Reactivate a historical version WITHOUT reprocessing (D-079).

        Steps:
        1. Validate the target (exists, superseded, fully indexed)
        2. DB status handshake (supersede current active → activate target)
        3. Vector payload flip (is_active_version flags)
        4. Index-state bump (cache correctness for Phase 18)
        """
        versions = DocumentVersionRepository(session)
        chunks = DocumentChunkRepository(session)
        documents = DocumentRepository(session)

        version = await versions.get_by_number(document_id, version_number)
        if version is None:
            raise DocumentNotFoundError(
                f"Version {version_number} of document {document_id} not found"
            )

        status = version.status.value if hasattr(version.status, "value") else str(version.status)
        if status == VersionStatus.ACTIVE.value:
            raise ConflictError(f"Version {version_number} is already the active version")
        if status != VersionStatus.SUPERSEDED.value:
            raise ConflictError(
                f"Version {version_number} cannot be reactivated from status '{status}'"
            )

        not_indexed = await chunks.count_not_indexed_for_version(version.id)
        if not_indexed > 0:
            raise ConflictError(
                f"Version {version_number} has {not_indexed} unindexed chunks; "
                "re-index before rollback"
            )

        # ---- transactional status handshake (Phase 3 invariant holds) --------
        await versions.activate(version.id)
        document = await documents.get_by_id(document_id)
        if document is not None:
            document.content_hash = version.content_hash
        await session.flush()

        # ---- vector layer flip + cache-state bump -----------------------------
        await self._indexer.switch_active_version(document_id, version.id)
        new_state = self._index_state.bump(str(document_id))

        logger.info(
            "version_rollback_completed",
            document_id=str(document_id),
            version=version_number,
            index_state=new_state,
        )
        return version