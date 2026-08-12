"""End-to-end ingestion tests against real PostgreSQL.

Covers: submission dedup, job execution, version creation + activation,
change detection on re-ingest, and citation metadata on chunk rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.models.enums import DocumentStatus, VersionStatus
from app.repositories.database import Database
from app.repositories.document import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
)
from app.repositories.user import UserRepository
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from tests.fixtures.builders import SAMPLE_MARKDOWN, build_pdf

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def services(database: Database, tmp_path: Path):
    settings = get_settings()
    chunking_registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    file_store = LocalFileStore(tmp_path / "uploads")
    pipeline = IngestionPipeline(build_default_registry(), chunking_registry, settings)
    service = IngestionService(file_store, chunking_registry, settings)
    runner = IngestionJobRunner(database, pipeline, file_store, settings)
    return service, runner


async def _owner(database: Database):
    async with database.session() as session:
        user = await UserRepository(session).create(email="ingest@example.com", hashed_password="x")
        await session.commit()
        return user


async def test_file_submission_is_idempotent(database: Database, services) -> None:
    service, _runner = services
    user = await _owner(database)
    content = build_pdf([[("Refund Policy", 18, True), ("Enterprise refunds within 30 days.", 11, False)]])

    async with database.session() as session:
        job1 = await service.submit_file(session=session, owner_id=user.id, filename="policy.pdf", content=content)
        job2 = await service.submit_file(session=session, owner_id=user.id, filename="policy.pdf", content=content)
        await session.commit()
    assert job1.id == job2.id  # deduplicated by idempotency key


async def test_full_ingestion_creates_active_version_with_citable_chunks(
    database: Database, services
) -> None:
    service, runner = services
    user = await _owner(database)

    async with database.session() as session:
        job = await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()

    assert await runner.run_next() is True

    async with database.session() as session:
        documents, versions_repo, chunks_repo = (
            DocumentRepository(session),
            DocumentVersionRepository(session),
            DocumentChunkRepository(session),
        )
        document = await documents.get_by_id(job.document_id)
        assert document is not None
        assert document.status == DocumentStatus.ACTIVE

        active = await versions_repo.get_active_for_document(document.id)
        assert active is not None
        assert active.version_number == 1
        assert active.status == VersionStatus.ACTIVE
        assert active.chunk_count and active.chunk_count > 0

        chunks = await chunks_repo.list_for_version(active.id, batch_size=100)
        assert len(chunks) == active.chunk_count
        assert all(c.qdrant_point_id for c in chunks)
        assert all(c.is_indexed is False for c in chunks)
        sections = {c.section for c in chunks if c.section}
        assert "Conditions" in sections  # citation-grade metadata survived the pipeline


async def test_reingest_changed_content_creates_v2_and_supersedes_v1(
    database: Database, services
) -> None:
    service, runner = services
    user = await _owner(database)

    async with database.session() as session:
        job = await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()
    await runner.run_next()

    changed = SAMPLE_MARKDOWN + "\n\n## Updates\n\nThe policy changed in 2026.\n"
    async with database.session() as session:
        docs = DocumentRepository(session)
        document = await docs.get_by_id(job.document_id)
        # same content hash → submit returns the existing job, so write the new
        # bytes through the store and enqueue via a fresh submission key path:
        from app.ingestion.domain import sha256_hex
        from app.ingestion.idempotency import compute_idempotency_key
        from app.repositories.job import IngestionJobRepository

        service.file_store.save(str(document.id), changed.encode())
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
        all_versions = await versions_repo.list_for_document(job.document_id)
        assert [v.version_number for v in all_versions] == [2, 1]
        statuses = {v.version_number: v.status for v in all_versions}
        assert statuses[2] == VersionStatus.ACTIVE
        assert statuses[1] == VersionStatus.SUPERSEDED