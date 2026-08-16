"""EmbeddingPipeline — the DB-aware embedding stage.

Loads chunks for a document version in batches, embeds them through the
EmbeddingService, and stamps the version row with the exact model identity
(embedding_model + embedding_model_version). Emits EmbeddedChunks for the
vector index; the Qdrant write itself is Phase 7's vector repository.

This same stage powers re-indexing after a model change: point it at any
version with a service built for the new model.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.embeddings.base import EmbeddedChunk
from app.embeddings.service import EmbeddingService
from app.repositories.document import DocumentChunkRepository, DocumentVersionRepository

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 256


class EmbeddingPipeline:
    def __init__(self, service: EmbeddingService) -> None:
        self._service = service

    async def embed_version(
        self,
        session: AsyncSession,
        version_id: uuid.UUID,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[EmbeddedChunk]:
        versions = DocumentVersionRepository(session)
        chunks_repo = DocumentChunkRepository(session)

        version = await versions.get_by_id(version_id)
        if version is None:
            raise DocumentNotFoundError(f"Document version {version_id} not found")

        embedded: list[EmbeddedChunk] = []
        offset = 0
        while True:
            batch = await chunks_repo.list_for_version(
                version_id, batch_size=batch_size, offset=offset
            )
            if not batch:
                break
            vectors = await self._service.embed_documents([chunk.content for chunk in batch])
            for chunk, vector in zip(batch, vectors):
                if chunk.qdrant_point_id is None:
                    raise ValueError(f"Chunk {chunk.id} has no deterministic point id")
                embedded.append(
                    EmbeddedChunk(
                        chunk_id=chunk.id,
                        point_id=chunk.qdrant_point_id,
                        vector=vector,
                        token_count=chunk.token_count or 0,
                    )
                )
            offset += len(batch)
            if len(batch) < batch_size:
                break

        # Model identity travels with the vectors (ARCHITECTURE.md §5.2).
        version.embedding_model = self._service.model_key
        version.embedding_model_version = self._service.model_version
        await session.flush()

        logger.info(
            "embedding_stage_completed",
            version_id=str(version_id),
            model=self._service.model_key,
            model_version=self._service.model_version,
            chunks=len(embedded),
        )
        return embedded