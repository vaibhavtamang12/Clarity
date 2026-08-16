"""Integration test: real PostgreSQL + deterministic model.

Verifies the Phase 6 stage end-to-end: persisted chunks → batched embedding →
EmbeddedChunks aligned with deterministic point ids → version row stamped
with exact model identity.
"""

from __future__ import annotations

import pytest

from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.factory import build_embedding_model
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.core.config import EmbeddingSettings
from app.models.document import DocumentChunk
from app.repositories.database import Database
from app.repositories.document import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
)
from app.repositories.user import UserRepository

pytestmark = pytest.mark.integration


async def test_embed_version_produces_aligned_vectors_and_stamps_identity(
    database: Database,
) -> None:
    registry = load_embedding_registry("configs/embeddings.yaml")
    model = build_embedding_model(registry, EmbeddingSettings(), model_key="hash_64")
    service = EmbeddingService(model, cache=InMemoryCacheStore(), max_retries=1)
    pipeline = EmbeddingPipeline(service)

    async with database.session() as session:
        user = await UserRepository(session).create(email="emb@example.com", hashed_password="x")
        doc = await DocumentRepository(session).create(
            owner_id=user.id, source_type="markdown", title="Embedding Test"
        )
        version = await DocumentVersionRepository(session).create(
            doc.id, 1, content_hash="h-emb", chunking_config={"strategy": "test"}
        )
        chunks = [
            DocumentChunk(document_id=doc.id, version_id=version.id, chunk_index=i, content=text)
            for i, text in enumerate(
                ["Refunds take 30 days.", "Passwords need 14 characters.", "Logs kept 365 days."]
            )
        ]
        await DocumentChunkRepository(session).bulk_create(chunks)
        await session.commit()
        version_id = version.id

    async with database.session() as session:
        embedded = await pipeline.embed_version(session, version_id, batch_size=2)
        await session.commit()

        assert len(embedded) == 3
        assert all(len(e.vector) == 64 for e in embedded)

        # Point ids must match the deterministic ids minted at ingestion.
        stored = await DocumentChunkRepository(session).list_for_version(version_id, batch_size=10)
        expected_points = {c.id: c.qdrant_point_id for c in stored}
        for e in embedded:
            assert e.point_id == expected_points[e.chunk_id]

        # Version row carries exact model identity (re-index traceability).
        refreshed = await DocumentVersionRepository(session).get_by_id(version_id)
        assert refreshed is not None
        assert refreshed.embedding_model == "hash_64"
        assert refreshed.embedding_model_version == "1"