"""Reranking over the REAL production sparse path (PostgreSQL tsvector)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.reranking.hash_reranker import HashReranker
from app.reranking.retriever import RerankedRetriever
from app.repositories.database import Database
from app.repositories.user import UserRepository
from app.retrieval.sparse import PostgresSparseRetriever
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from tests.fixtures.builders import SAMPLE_MARKDOWN

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


async def test_reranked_sparse_pipeline(database: Database, tmp_path: Path) -> None:
    settings = get_settings()
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    store = LocalFileStore(tmp_path / "up")
    service = IngestionService(store, registry, settings)
    runner = IngestionJobRunner(
        database, IngestionPipeline(build_default_registry(), registry, settings), store, settings
    )
    async with database.session() as session:
        user = await UserRepository(session).create(email="rerank@example.com", hashed_password="x")
        await service.submit_file(
            session=session, owner_id=user.id, filename="policy.md", content=SAMPLE_MARKDOWN.encode()
        )
        await session.commit()
    assert await runner.run_next() is True

    retriever = RerankedRetriever(
        PostgresSparseRetriever(database), HashReranker(), candidates=20, top_n=5
    )
    result = await retriever.retrieve("enterprise refund window days", top_k=5)

    assert result.metadata.retriever == "reranked"
    assert result.metadata.strategy.endswith("+rerank")
    assert result.metadata.counts["rerank_candidates"] > 0
    assert len(result.items) <= 5
    assert all(item.rerank_score is not None for item in result.items)
    scores = [item.rerank_score for item in result.items]
    assert scores == sorted(scores, reverse=True)
    assert any("30-day refund window" in item.content for item in result.items)