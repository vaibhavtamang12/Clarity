"""Semantic tests for the vector repository contract (in-memory implementation).

These define the behavior Qdrant must match — verified against the real
server in tests/integration/test_qdrant_repository.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.repositories.vector.base import VectorFilter, VectorPayload, VectorPoint
from app.repositories.vector.in_memory_repository import InMemoryVectorRepository


def _payload(**overrides: object) -> VectorPayload:
    base = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "version_id": uuid.uuid4(),
        "owner_id": uuid.uuid4(),
        "is_active_version": True,
        "source_type": "markdown",
        "content": "chunk text",
        "token_count": 10,
        "created_at": datetime.now(timezone.utc),
    }
    base.update(overrides)
    return VectorPayload(**base)  # type: ignore[arg-type]


def _point(vector: list[float], **payload_overrides: object) -> VectorPoint:
    return VectorPoint(point_id=uuid.uuid4(), vector=vector, payload=_payload(**payload_overrides))


@pytest.mark.asyncio
async def test_search_ranks_by_cosine_similarity() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(3)
    target = _point([1.0, 0.0, 0.0], content="north")
    await repo.upsert_points([
        target,
        _point([0.0, 1.0, 0.0], content="east"),
        _point([0.7, 0.7, 0.0], content="northeast"),
    ])
    results = await repo.search([1.0, 0.0, 0.0], top_k=3)
    assert [r.content for r in results][0] == "north"
    assert results[0].score > results[1].score > results[2].score


@pytest.mark.asyncio
async def test_upsert_is_idempotent_by_point_id() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(2)
    point_id = uuid.uuid4()
    p1 = VectorPoint(point_id=point_id, vector=[1.0, 0.0], payload=_payload(content="v1"))
    p2 = VectorPoint(point_id=point_id, vector=[0.0, 1.0], payload=_payload(content="v2"))
    await repo.upsert_points([p1])
    await repo.upsert_points([p2])
    assert await repo.count() == 1
    results = await repo.search([0.0, 1.0], top_k=1)
    assert results[0].content == "v2"  # replaced, not duplicated


@pytest.mark.asyncio
async def test_filters_document_tags_active_and_dates() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(2)
    doc_a = uuid.uuid4()
    now = datetime.now(timezone.utc)

    await repo.upsert_points([
        _point([1.0, 0.0], document_id=doc_a, tags=("legal", "billing"), is_active_version=True, created_at=now),
        _point([0.9, 0.1], document_id=doc_a, tags=("hr",), is_active_version=False, created_at=now - timedelta(days=40)),
        _point([0.0, 1.0], document_id=uuid.uuid4(), tags=("legal",), is_active_version=True, created_at=now),
    ])

    assert await repo.count(VectorFilter(document_id=doc_a)) == 2
    assert await repo.count(VectorFilter(tags=("hr",))) == 1
    assert await repo.count(VectorFilter(is_active_version=True)) == 2
    assert await repo.count(VectorFilter(created_after=now - timedelta(days=30))) == 2
    results = await repo.search([1.0, 0.0], top_k=5, filter_=VectorFilter(document_id=doc_a, tags=("legal",)))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_delete_points_and_delete_by_filter() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(2)
    doc = uuid.uuid4()
    keep = _point([1.0, 0.0], document_id=uuid.uuid4())
    doomed = _point([0.0, 1.0], document_id=doc)
    await repo.upsert_points([keep, doomed])

    await repo.delete_by_filter(VectorFilter(document_id=doc))
    assert await repo.count() == 1

    await repo.delete_points([keep.point_id])
    assert await repo.count() == 0


@pytest.mark.asyncio
async def test_update_payload_by_filter_flips_active_flag() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(2)
    version = uuid.uuid4()
    await repo.upsert_points([_point([1.0, 0.0], version_id=version, is_active_version=False)])

    await repo.update_payload_by_filter(VectorFilter(version_id=version), {"is_active_version": True})

    assert await repo.count(VectorFilter(version_id=version, is_active_version=True)) == 1


@pytest.mark.asyncio
async def test_score_threshold_filters_results() -> None:
    repo = InMemoryVectorRepository()
    await repo.ensure_collection(2)
    await repo.upsert_points([
        _point([1.0, 0.0], content="high"),
        _point([0.0, 1.0], content="low"),
    ])
    results = await repo.search([1.0, 0.0], top_k=5, score_threshold=0.5)
    assert [r.content for r in results] == ["high"]