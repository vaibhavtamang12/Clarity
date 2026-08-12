"""The ingestion pipeline: parse → clean → structure → metadata → chunk → persist.

Embedding and vector indexing attach in Phases 6–7 at the marked hook point;
everything up to persisted, citation-ready chunks is complete here.

Behaviors guaranteed:
- Idempotent: identical content_hash for the latest version → no new version.
- Versioned: every run creates version N+1 with frozen pipeline config.
- Citation-ready: every chunk row carries page/section/heading_path + a
  deterministic Qdrant point id.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.chunking.factory import build_chunker
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.cleaning import clean_parsed_document
from app.ingestion.domain import sha256_hex
from app.ingestion.metadata import extract_metadata
from app.ingestion.parsers.base import ParserRegistry
from app.ingestion.structure import annotate_sections
from app.models.document import Document, DocumentChunk
from app.models.enums import DocumentStatus
from app.repositories.document import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
)
from app.utils.ids import chunk_point_id

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    document_id: uuid.UUID
    version_id: uuid.UUID | None
    version_number: int | None
    chunk_count: int
    content_hash: str
    changed: bool
    title: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        parsers: ParserRegistry,
        chunking_registry: ChunkingRegistry,
        settings: Settings,
    ) -> None:
        self.parsers = parsers
        self.chunking_registry = chunking_registry
        self.settings = settings

    async def ingest(
        self,
        *,
        session: AsyncSession,
        document: Document,
        content: bytes,
    ) -> IngestionResult:
        content_hash = sha256_hex(content)
        documents = DocumentRepository(session)
        versions = DocumentVersionRepository(session)
        chunks_repo = DocumentChunkRepository(session)

        # ---- change detection (idempotency at the version level) ----------
        latest = await versions.get_latest_for_document(document.id)
        if latest is not None and latest.content_hash == content_hash:
            logger.info(
                "ingestion_skipped_unchanged",
                document_id=str(document.id),
                version=latest.version_number,
            )
            return IngestionResult(
                document_id=document.id,
                version_id=latest.id,
                version_number=latest.version_number,
                chunk_count=latest.chunk_count or 0,
                content_hash=content_hash,
                changed=False,
            )

        # ---- parse → clean → structure → metadata → chunk ------------------
        parser = self.parsers.get(document.source_type)
        try:
            parsed = parser.parse(content, source_uri=document.source_uri or "")
        except Exception as exc:  # noqa: BLE001 — normalize to IngestionError for job retries
            raise IngestionError(
                f"Parsing failed for document {document.id}: {type(exc).__name__}"
            ) from exc

        cleaned = clean_parsed_document(parsed)
        structured = annotate_sections(cleaned)
        metadata = extract_metadata(structured, fallback_title=document.title)

        strategy_name = self.chunking_registry.default
        strategy_config = self.chunking_registry.strategies[strategy_name]
        chunker = build_chunker(strategy_config)
        chunks = chunker.chunk(structured.blocks)
        if not chunks:
            raise IngestionError(f"No chunkable content produced for document {document.id}")

        # ---- persist version + chunks ---------------------------------------
        version_number = (latest.version_number + 1) if latest else 1
        version = await versions.create(
            document.id,
            version_number,
            content_hash=content_hash,
            chunking_config=strategy_config.model_dump(),
            embedding_model=self.settings.embedding.default_model,
            parser=parser.name,
            mime_type=None,
            file_size_bytes=len(content),
            page_count=metadata.page_count,
            token_count=sum(c.token_count for c in chunks),
            chunk_count=len(chunks),
        )

        rows = [
            DocumentChunk(
                document_id=document.id,
                version_id=version.id,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                heading_path=chunk.heading_path or None,
                qdrant_point_id=chunk_point_id(version.id, chunk.chunk_index),
                is_indexed=False,  # flipped by the indexing stage (Phase 7)
                metadata_={"source_uri": document.source_uri, "title": metadata.title},
            )
            for chunk in chunks
        ]
        await chunks_repo.bulk_create(rows)

        # ---- activate version + finalize document ---------------------------
        await versions.activate(version.id)
        document.status = DocumentStatus.ACTIVE
        document.content_hash = content_hash
        if metadata.title:
            document.title = metadata.title
        await session.flush()

        logger.info(
            "ingestion_completed",
            document_id=str(document.id),
            version=version_number,
            chunks=len(chunks),
        )
        return IngestionResult(
            document_id=document.id,
            version_id=version.id,
            version_number=version_number,
            chunk_count=len(chunks),
            content_hash=content_hash,
            changed=True,
            title=metadata.title,
        )