"""Document, DocumentVersion, and DocumentChunk models (ARCHITECTURE.md §5.1).

Versioning invariants:
- One document → many versions; exactly ONE version may be 'active'.
  This is enforced at the database level by a partial unique index
  (uq_document_versions_one_active), not by application optimism.
- Chunks of all versions are retained; retrieval filters by active version.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    Base,
    CreatedAtMixin,
    StrEnumColumn,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import DocumentSourceType, DocumentStatus, VersionStatus


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(512))
    source_type: Mapped[DocumentSourceType] = mapped_column(
        StrEnumColumn(DocumentSourceType), nullable=False
    )
    source_uri: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        StrEnumColumn(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )
    # NOTE: attribute is `metadata_` because DeclarativeBase reserves `.metadata`;
    # the actual column name in PostgreSQL is "metadata".
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_owner_status", "owner_id", "status"),)


class DocumentVersion(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(100))
    parser: Mapped[str | None] = mapped_column(String(50))
    # Frozen pipeline configuration — every chunk of this version is
    # reproducible with exactly these settings.
    chunking_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedding_model_version: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[VersionStatus] = mapped_column(
        StrEnumColumn(VersionStatus),
        nullable=False,
        default=VersionStatus.PROCESSING,
        server_default=VersionStatus.PROCESSING.value,
    )
    page_count: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)

    document: Mapped["Document"] = relationship(back_populates="versions")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id", "version_number", name="uq_document_versions_document_version"
        ),
        # Exactly one active version per document — DB-enforced.
        Index(
            "uq_document_versions_one_active",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )


class DocumentChunk(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Generated full-text search vector: the sparse (BM25-style) index.
    # Kept in PostgreSQL initially per ADR / decision D-010.
    content_tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True),
        nullable=False,
    )
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    token_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int | None] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(512))
    heading_path: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    qdrant_point_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    is_indexed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
    version: Mapped["DocumentVersion"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "version_id", "chunk_index", name="uq_document_chunks_version_chunk"
        ),
        Index("ix_document_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_document_chunks_document_version", "document_id", "version_id"),
    )