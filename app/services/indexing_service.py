"""Indexing stage — completes the ingestion pipeline (ARCHITECTURE.md §4).

    chunks (PG) → embed (Phase 6) → upsert (Phase 7) → is_indexed=True
                                    → active-version payload flags

Also owns the version-switch payload flip (decision D-040): activating v2 is
a metadata update on existing points — never a re-embedding.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.embeddings.pipeline import EmbeddingPipeline
from app.models.document import Document, DocumentChunk, DocumentVersion
from app.models.enums import VersionStatus
from app.repositories.vector.base import VectorFilter, VectorPayload, VectorPoint, VectorRepository

logger = get_logger(__name__)


def build_payload(
    *,
    chunk: DocumentChunk,
    document: Document,
    version: DocumentVersion,
    is_active: bool,
) -> VectorPayload:
    """Assemble the point payload from the system-of-record rows."""
    meta = document.metadata_ or {}
    source_type = document.source_type.value if hasattr(document.source_type, "value") else str(document.source_type)
    return VectorPayload(
        chunk_id=chunk.id,
        document_id=document.id,
        version_id=version.id,
        owner_id=document.owner_id,
        is_active_version=is_active,
        source_type=source_type,
        source_uri=document.source_uri,
        title=document.title,
        section=chunk.section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        tags=tuple(meta.get("tags") or ()),
        department=meta.get("department"),
        document_type=meta.get("document_type") or source_type,
        token_count=chunk.token_count or 0,
        content=chunk.content,
        created_at=chunk.created_at or datetime.now(timezone.utc),
    )


class IndexingService:
    def __init__(
        self,
        repository: VectorRepository,
        embedding_pipeline: EmbeddingPipeline,
        dimension: int,
        index_state: IndexStateNotifier | None = None,   # Phase 18 addition
    ) -> None:
        self._repository = repository
        self._embedding = embedding_pipeline
        self._dimension = dimension
        self._index_state = index_state
    
    async def index_version(self, session: AsyncSession, version_id: uuid.UUID) -> int:
        version = await session.get(DocumentVersion, version_id)
        if version is None:
            raise DocumentNotFoundError(f"Document version {version_id} not found")
        document = await session.get(Document, version.document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {version.document_id} not found")
        if self._index_state is not None:
            await self._index_state.bump(self._repository.collection)

        # ---- embed (Phase 6 stage) ------------------------------------------
        embedded = await self._embedding.embed_version(session, version_id)

        # ---- load chunk rows for payload fields ------------------------------
        chunk_rows = (
            await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.version_id == version_id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
        ).scalars().all()
        chunks_by_id = {chunk.id: chunk for chunk in chunk_rows}

        # ---- build + upsert points -------------------------------------------
        is_active = version.status == VersionStatus.ACTIVE
        points = [
            VectorPoint(
                point_id=e.point_id,
                vector=e.vector,
                payload=build_payload(
                    chunk=chunks_by_id[e.chunk_id],
                    document=document,
                    version=version,
                    is_active=is_active,
                ),
            )
            for e in embedded
        ]
        await self._repository.ensure_collection(self._dimension)
        await self._repository.upsert_points(points)

        # ---- flip active-version flags across the whole document (D-040) -----
        if is_active:
            await self._repository.update_payload_by_filter(
                VectorFilter(document_id=document.id), {"is_active_version": False}
            )
            await self._repository.update_payload_by_filter(
                VectorFilter(version_id=version.id), {"is_active_version": True}
            )

        # ---- mark chunks indexed ----------------------------------------------
        await session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.version_id == version_id)
            .values(is_indexed=True)
        )
        await session.flush()

        logger.info(
            "indexing_completed",
            version_id=str(version_id),
            points=len(points),
            collection=self._repository.collection,
            is_active=is_active,
        )
        return len(points)
    
    async def switch_active_version(
        self, document_id: uuid.UUID, version_id: uuid.UUID
    ) -> None:
        """Flip is_active_version payload flags WITHOUT re-embedding (D-079).

        Historical vectors are preserved by design (ARCHITECTURE §5.2), so
        rollback and version switches are O(points) metadata operations.
        """
        await self._repository.update_payload_by_filter(
            VectorFilter(document_id=document_id), {"is_active_version": False}
        )
        await self._repository.update_payload_by_filter(
            VectorFilter(version_id=version_id), {"is_active_version": True}
        )
        logger.info(
            "active_version_switched",
            document_id=str(document_id),
            version_id=str(version_id),
            collection=self._repository.collection,
        )
        if self._index_state is not None:
            await self._index_state.bump(self._repository.collection)