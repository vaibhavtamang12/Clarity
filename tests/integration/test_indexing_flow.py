"""End-to-end: submit → job → parse/chunk → embed → index → dense search.

PostgreSQL is real; the vector layer uses the in-memory repository (Qdrant
contract semantics are covered separately). This test proves the FULL
ingestion-to-retrieval path, including the v2 version flip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import EmbeddingSettings, get_settings
from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.factory import build_embedding_model
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.domain import sha256_hex
from app.ingestion.idempotency import compute_idempotency_key
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.models.enums import VersionStatus
from app.retrieval.dense import DenseSearchService
from app.retrieval.dense import DenseRetrieve
from app.repositories.database import Database
from app.repositories.document import DocumentChunkRepository, DocumentRepository, DocumentVersionRepository
from app.repositories.job import IngestionJobRepository
from app.repositories.user import UserRepository
from app.repositories.vector.base import VectorFilter
from app.repositories.vector.in_memory_repository import InMemoryVectorRepository
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from app.services.indexing_service import IndexingService
from tests.fixtures.builders import SAMPLE_MARKDOWN

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def platform(database: Database, tmp_path: Path):
    settings = get_settings()
    chunking_registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    embedding_registry = load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")

    file_store = LocalFileStore(tmp_path / "uploads")
    pipeline = IngestionPipeline(build_default_registry(), chunking_registry, settings)

    model = build_embedding_model(embedding_registry, EmbeddingSettings(), model_key="hash_64")
    embedding_service = EmbeddingService(model, cache=InMemoryCacheStore(), max_retries=1)
    vector_repo = InMemoryVectorRepository()
    indexer = IndexingService(
        repository=vector_repo,
        embedding_pipeline=EmbeddingPipeline(embedding_service),
        dimension=model.dimension,
    )

    service = IngestionService(file_store, chunking_registry, settings)
    runner = IngestionJobRunner(database, pipeline, file_store, settings, indexer=indexer)
    dense = DenseRetriever(vector_repo, embedding_service)
    return service, runner, dense, vector_repo, file_store


async def _owner(database: Database):
    async with database.session() as session:
        user = await UserRepository(session).create(email="idx@example.com", hashed_password="x")
        await session.commit()
        return user


async def test_full_ingestion_to_search_with_version_flip(database: Database, platform) -> None:
    service, runner, dense, vector_repo, file_store = platform
    user = await _owner(database)

    # ---- v1 ---------------------------------------------------------------
    async with database.session() as session:
        job = await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()
    assert await runner.run_next() is True

    async with database.session() as session:
        chunks = await DocumentChunkRepository(session).list_for_version(
            (await DocumentVersionRepository(session).get_active_for_document(job.document_id)).id,
            batch_size=100,
        )
        assert all(c.is_indexed for c in chunks)
        v1_id = (await DocumentVersionRepository(session).get_active_for_document(job.document_id)).id

    assert await vector_repo.count(VectorFilter(is_active_version=True)) > 0

    results = await dense.search("refund window for enterprise customers", top_k=5)
    assert results
    assert any("30-day refund window" in r.content for r in results)
    assert results[0].page_start is not None or results[0].section  # citation metadata alive

    # ---- v2 (changed content) ----------------------------------------------
    changed = SAMPLE_MARKDOWN + "\n\n## Updates\n\nThe enterprise refund window changed to 45 days in 2026.\n"
    async with database.session() as session:
        document = await DocumentRepository(session).get_by_id(job.document_id)
        file_store.save(str(document.id), changed.encode())
        settings = get_settings()
        registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
        strategy = registry.strategies[registry.default]
        key = compute_idempotency_key(
            content_hash=sha256_hex(changed.encode()),
            parser="markdown",
            chunking_strategy=registry.default,
            chunking_config=strategy.model_dump(mode="json"),
            embedding_model=settings.embedding.default_model,
        )
        jobs = IngestionJobRepository(session)
        new_job = await jobs.create(document.id, idempotency_key=key)
        await jobs.enqueue(new_job)
        await session.commit()

    assert await runner.run_next() is True

    async with database.session() as session:
        versions_repo = DocumentVersionRepository(session)
        v2 = await versions_repo.get_active_for_document(job.document_id)
        assert v2 is not None and v2.version_number == 2

    # ---- version-aware retrieval invariants ---------------------------------
    # Default search: ONLY v2 points are visible.
    default_results = await dense.search("enterprise refund window", top_k=20)
    assert default_results
    assert all(r.version_id == v2.id for r in default_results)
    assert any("45 days" in r.content for r in default_results)

    # Explicit v1 filter: history remains queryable (FR-10.3).
    v1_results = await dense.search(
        "enterprise refund window", top_k=20, filter_=VectorFilter(version_id=v1_id)
    )
    assert v1_results
    assert all(r.version_id == v1_id for r in v1_results)