"""Dense search semantics: active-version default, explicit overrides."""

from __future__ import annotations

import uuid

import pytest

from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.hash_model import HashEmbeddingModel
from app.embeddings.service import EmbeddingService
from app.retrieval.dense import DenseSearchService
from app.retrieval.dense import DenseRetriever
from app.repositories.vector.base import VectorFilter, VectorPayload, VectorPoint
from app.repositories.vector.in_memory_repository import InMemoryVectorRepository


@pytest.fixture()
def stack():
    model = HashEmbeddingModel(dimension=64)
    service = EmbeddingService(model, cache=InMemoryCacheStore())
    repo = InMemoryVectorRepository()
    return repo, DenseSearchService(repo, service), model


async def _seed(repo, model, *, version_id: uuid.UUID, active: bool, text: str):
    vector = model.embed_query(text)
    payload = VectorPayload(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=version_id,
        owner_id=uuid.uuid4(), is_active_version=active, source_type="markdown",
        content=text, token_count=10,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    await repo.upsert_points([VectorPoint(point_id=uuid.uuid4(), vector=vector, payload=payload)])


@pytest.mark.asyncio
async def test_default_search_returns_only_active_version(stack) -> None:
    repo, dense, model = stack
    v1, v2 = uuid.uuid4(), uuid.uuid4()
    await _seed(repo, model, version_id=v1, active=False, text="old refund policy says 14 days")
    await _seed(repo, model, version_id=v2, active=True, text="new refund policy says 30 days")

    results = await dense.search("refund policy", top_k=10)
    assert len(results) == 1
    assert results[0].version_id == v2


@pytest.mark.asyncio
async def test_explicit_version_filter_overrides_active_default(stack) -> None:
    repo, dense, model = stack
    v1, v2 = uuid.uuid4(), uuid.uuid4()
    await _seed(repo, model, version_id=v1, active=False, text="old refund policy says 14 days")
    await _seed(repo, model, version_id=v2, active=True, text="new refund policy says 30 days")

    results = await dense.search("refund policy", top_k=10, filter_=VectorFilter(version_id=v1))
    assert len(results) == 1
    assert results[0].version_id == v1


@pytest.mark.asyncio
async def test_relevance_ordering_end_to_end(stack) -> None:
    repo, dense, model = stack
    vid = uuid.uuid4()
    await _seed(repo, model, version_id=vid, active=True, text="password requirements need 14 characters minimum")
    await _seed(repo, model, version_id=vid, active=True, text="enterprise refund window is 30 days from invoice")

    results = await dense.search("enterprise refund window", top_k=2)
    assert len(results) == 2
    assert "refund" in results[0].content

@pytest.fixture()
def stack():
    model = HashEmbeddingModel(dimension=64)
    service = EmbeddingService(model, cache=InMemoryCacheStore())
    repo = InMemoryVectorRepository()
    return repo, DenseRetriever(repo, service), model

# Inside each test, `dense.search(...)` becomes:
#   (await dense.search(...))          →  (await dense.retrieve(...)).items
# e.g.: