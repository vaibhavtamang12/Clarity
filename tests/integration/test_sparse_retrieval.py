"""Production sparse path: PostgreSQL tsvector + GIN index (Phase 3's payoff)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.repositories.database import Database
from app.repositories.user import UserRepository
from app.repositories.vector.base import VectorFilter
from app.retrieval.sparse import PostgresSparseRetriever
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from tests.fixtures.builders import SAMPLE_MARKDOWN

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
async def ingested_doc(database: Database, tmp_path: Path):
    settings = get_settings()
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    service = IngestionService(LocalFileStore(tmp_path / "up"), registry, settings)
    runner = IngestionJobRunner(
        database, IngestionPipeline(build_default_registry(), registry, settings),
        LocalFileStore(tmp_path / "up"), settings,
    )
    async with database.session() as session:
        user = await UserRepository(session).create(email="sparse@example.com", hashed_password="x")
        job = await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()
    assert await runner.run_next() is True
    return job.document_id, user.id


async def test_sparse_retrieval_finds_expected_chunk(database: Database, ingested_doc) -> None:
    document_id, owner_id = ingested_doc
    retriever = PostgresSparseRetriever(database)

    result = await retriever.retrieve("enterprise refund window days", top_k=5)
    assert result.metadata.retriever == "sparse_postgres"
    assert result.items
    assert any("30-day refund window" in item.content for item in result.items)
    assert all(item.version_id for item in result.items)


async def test_sparse_respects_document_and_owner_filters(database: Database, ingested_doc) -> None:
    document_id, owner_id = ingested_doc
    retriever = PostgresSparseRetriever(database)

    scoped = await retriever.retrieve(
        "refund", top_k=5, filter_=VectorFilter(document_id=document_id)
    )
    assert scoped.items
    assert all(item.document_id == document_id for item in scoped.items)

    foreign = await retriever.retrieve(
        "refund", top_k=5, filter_=VectorFilter(owner_id=uuid.uuid4())
    )
    assert foreign.items == []


async def test_unsupported_filters_raise_explicitly(database: Database, ingested_doc) -> None:
    retriever = PostgresSparseRetriever(database)
    with pytest.raises(ValueError, match="department"):
        await retriever.retrieve("refund", filter_=VectorFilter(department="legal"))