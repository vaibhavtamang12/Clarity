"""Hybrid over REAL PostgreSQL sparse + in-memory dense, including degradation."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.core.config import RetrievalSettings, get_settings
from app.core.exceptions import RetrievalUnavailableError
from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.factory import build_embedding_model
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.models.document import DocumentChunk
from app.repositories.database import Database
from app.repositories.document import DocumentChunkRepository, DocumentVersionRepository
from app.repositories.user import UserRepository
from app.repositories.vector.base import VectorPayload, VectorPoint
from app.repositories.vector.in_memory_repository import InMemoryVectorRepository
from app.retrieval.base import RetrievalMetadata, RetrievalResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import PostgresSparseRetriever
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from tests.fixtures.builders import SAMPLE_MARKDOWN

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


class BrokenDense:
    name = "dense"

    async def retrieve(self, query: str, *, top_k: int = 10, filter_=None) -> RetrievalResult:
        raise RuntimeError("vector store down")


async def _ingest(database: Database, tmp_path: Path):
    settings = get_settings()
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    store = LocalFileStore(tmp_path / "up")
    service = IngestionService(store, registry, settings)
    runner = IngestionJobRunner(
        database, IngestionPipeline(build_default_registry(), registry, settings), store, settings
    )
    async with database.session() as session:
        user = await UserRepository(session).create(email="hybrid@example.com", hashed_password="x")
        job = await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()
    assert await runner.run_next() is True
    return job.document_id


async def _dense_over(database: Database, document_id: uuid.UUID) -> DenseRetriever:
    """Mirror persisted chunks into the in-memory vector repo (hash model)."""
    embedding_registry = load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")
    model = build_embedding_model(embedding_registry, get_settings().embedding, model_key="hash_64")
    service = EmbeddingService(model, cache=InMemoryCacheStore())
    repo = InMemoryVectorRepository(collection="hybrid-test")

    async with database.session() as session:
        version = await DocumentVersionRepository(session).get_active_for_document(document_id)
        chunks = await DocumentChunkRepository(session).list_for_version(version.id, batch_size=200)

    vectors = await service.embed_documents([c.content for c in chunks])
    from datetime import datetime, timezone

    points = [
        VectorPoint(
            point_id=chunk.qdrant_point_id,
            vector=vector,
            payload=VectorPayload(
                chunk_id=chunk.id, document_id=chunk.document_id, version_id=chunk.version_id,
                owner_id=uuid.uuid4(), is_active_version=True, source_type="markdown",
                content=chunk.content, token_count=chunk.token_count or 0,
                created_at=datetime.now(timezone.utc), section=chunk.section,
            ),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    await repo.upsert_points(points)
    return DenseRetriever(repo, service)


async def test_hybrid_combines_real_sparse_and_dense(database: Database, tmp_path: Path) -> None:
    document_id = await _ingest(database, tmp_path)
    settings = RetrievalSettings(
        dense_top_k=20, sparse_top_k=20, final_top_k=5,
        fusion_strategy="rrf", degrade_policy="strict",  # type: ignore[arg-type]
    )
    hybrid = HybridRetriever(
        await _dense_over(database, document_id), PostgresSparseRetriever(database), settings
    )

    result = await hybrid.retrieve("enterprise refund window", top_k=5)
    assert result.metadata.degraded is False
    assert result.metadata.strategy == "rrf"
    assert result.metadata.counts["dense"] > 0
    assert result.metadata.counts["sparse"] > 0
    assert result.items
    assert any("30-day refund window" in item.content for item in result.items)
    multi_source = [item for item in result.items if len(item.sources) > 1]
    assert multi_source, "expected at least one chunk found by BOTH branches"


async def test_hybrid_degrades_when_dense_is_down(database: Database, tmp_path: Path) -> None:
    document_id = await _ingest(database, tmp_path)
    settings = RetrievalSettings(degrade_policy="degrade")  # type: ignore[call-arg]
    hybrid = HybridRetriever(BrokenDense(), PostgresSparseRetriever(database), settings)

    result = await hybrid.retrieve("enterprise refund window", top_k=5)
    assert result.metadata.degraded is True
    assert "dense" in (result.metadata.degraded_reason or "")
    assert result.items  # sparse-only, still useful

    strict = RetrievalSettings(degrade_policy="strict")  # type: ignore[call-arg]
    hybrid_strict = HybridRetriever(BrokenDense(), PostgresSparseRetriever(database), strict)
    with pytest.raises(RetrievalUnavailableError):
        await hybrid_strict.retrieve("enterprise refund window")