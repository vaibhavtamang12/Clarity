"""Repositories for Document, DocumentVersion, DocumentChunk.

The version-activation handshake lives here: it is a single flush-scoped
sequence (supersede current active → mark new active) that callers commit.
The partial unique index guarantees the invariant even if two callers race.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select, update

from app.core.exceptions import DocumentNotFoundError
from app.models.document import Document, DocumentChunk, DocumentVersion
from app.models.enums import DocumentStatus, VersionStatus
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def create(
        self,
        owner_id: uuid.UUID,
        source_type: str,
        title: str | None = None,
        source_uri: str | None = None,
    ) -> Document:
        document = Document(
            owner_id=owner_id,
            source_type=source_type,  # type: ignore[arg-type]
            title=title,
            source_uri=source_uri,
        )
        return await self.add(document)

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.owner_id == owner_id, Document.status != DocumentStatus.DELETED)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def get_by_content_hash(self, content_hash: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.content_hash == content_hash).limit(1)
        )
        return result.scalar_one_or_none()

    async def update_status(self, document_id: uuid.UUID, status: DocumentStatus) -> None:
        await self.session.execute(
            update(Document).where(Document.id == document_id).values(status=status)
        )
        await self.session.flush()


class DocumentVersionRepository(BaseRepository[DocumentVersion]):
    model = DocumentVersion

    async def create(
        self,
        document_id: uuid.UUID,
        version_number: int,
        content_hash: str,
        chunking_config: dict | None = None,
        embedding_model: str | None = None,
        **fields: object,
    ) -> DocumentVersion:
        version = DocumentVersion(
            document_id=document_id,
            version_number=version_number,
            content_hash=content_hash,
            chunking_config=chunking_config or {},
            embedding_model=embedding_model,
            **fields,  # type: ignore[arg-type]
        )
        return await self.add(version)

    async def get_active_for_document(self, document_id: uuid.UUID) -> DocumentVersion | None:
        result = await self.session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.status == VersionStatus.ACTIVE,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[DocumentVersion]:
        result = await self.session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        )
        return result.scalars().all()

    async def activate(self, version_id: uuid.UUID) -> DocumentVersion:
        """Transactional version handshake.

        Locks the target version row, supersedes any currently active version
        of the same document, then activates the target. Callers commit.
        """
        result = await self.session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .with_for_update()
        )
        version = result.scalar_one_or_none()
        if version is None:
            raise DocumentNotFoundError(f"Document version {version_id} not found")

        await self.session.execute(
            update(DocumentVersion)
            .where(
                DocumentVersion.document_id == version.document_id,
                DocumentVersion.status == VersionStatus.ACTIVE,
            )
            .values(status=VersionStatus.SUPERSEDED)
        )
        version.status = VersionStatus.ACTIVE
        await self.session.flush()
        return version


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    async def bulk_create(self, chunks: Sequence[DocumentChunk]) -> int:
        self.session.add_all(chunks)
        await self.session.flush()
        return len(chunks)

    async def count_for_version(self, version_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(DocumentChunk.id)
            .where(DocumentChunk.version_id == version_id)
            .with_only_fields(DocumentChunk.id)
        )
        return len(result.all())

    async def list_for_version(
        self, version_id: uuid.UUID, batch_size: int = 500, offset: int = 0
    ) -> Sequence[DocumentChunk]:
        """Deterministically ordered batch read — used by the embedding pipeline."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.version_id == version_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .limit(batch_size)
            .offset(offset)
        )
        return result.scalars().all()