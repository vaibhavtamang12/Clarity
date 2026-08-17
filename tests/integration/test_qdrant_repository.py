"""Contract verification against a REAL Qdrant server.

Same semantics as the in-memory suite — this is what proves the two
implementations are interchangeable (Rule 4).
"""

from __future__ import annotations

import uuid

import pytest

from app.repositories.vector.base import VectorFilter, VectorPayload, VectorPoint

pytestmark = pytest.mark.integration


def _point(vector: list[float], **overrides: object) -> VectorPoint:
    from datetime import datetime, timezone

    payload_kwargs = {
        "chunk_id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "version_id": uuid.uuid4(),
        "owner_id": uuid.uuid4(),
        "is_active_version": True,
        "source_type": "markdown",
        "content": overrides.pop("content", "text"),
        "token_count": 5,
        "created_at": datetime.now(timezone.utc),
        "tags": overrides.pop("tags", ()),
    }
    payload_kwargs.update(overrides)
    return VectorPoint(
        point_id=uuid.uuid4(), vector=vector, payload=VectorPayload(**payload_kwargs)  # type: ignore[arg-type]
    )


async def test_collection_lifecycle(qdrant_repo) -> None:
    assert await qdrant_repo.collection_exists() is False
    await qdrant_repo.ensure_collection(8)
    assert await qdrant_repo.collection_exists() is True
    # ensure_collection is idempotent
    await qdrant_repo.ensure_collection(8)


async def test_upsert_search_filter_count_delete(qdrant_repo) -> None:
    await qdrant_repo.ensure_collection(4)
    doc_a = uuid.uuid4()

    north = _point([1.0, 0.0, 0.0, 0.0], content="north", document_id=doc_a, tags=("geo",))
    east = _point([0.0, 1.0, 0.0, 0.0], content="east", document_id=doc_a)
    other = _point([0.0, 0.0, 1.0, 0.0], content="other", document_id=uuid.uuid4())
    assert await qdrant_repo.upsert_points([north, east, other]) == 3
    assert await qdrant_repo.count() == 3

    results = await qdrant_repo.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert results[0].content == "north"
    assert results[0].score > results[1].score

    filtered = await qdrant_repo.search(
        [1.0, 0.0, 0.0, 0.0], top_k=5, filter_=VectorFilter(document_id=doc_a)
    )
    assert {r.content for r in filtered} == {"north", "east"}

    tagged = await qdrant_repo.count(VectorFilter(tags=("geo",)))
    assert tagged == 1

    await qdrant_repo.delete_by_filter(VectorFilter(document_id=doc_a))
    assert await qdrant_repo.count() == 1

    await qdrant_repo.delete_points([other.point_id])
    assert await qdrant_repo.count() == 0


async def test_active_version_payload_flip(qdrant_repo) -> None:
    await qdrant_repo.ensure_collection(2)
    v1, v2 = uuid.uuid4(), uuid.uuid4()
    doc = uuid.uuid4()
    await qdrant_repo.upsert_points([
        _point([1.0, 0.0], content="v1 text", document_id=doc, version_id=v1, is_active_version=True),
        _point([0.9, 0.1], content="v2 text", document_id=doc, version_id=v2, is_active_version=False),
    ])

    # Simulate the version-switch handshake: deactivate document, activate v2.
    await qdrant_repo.update_payload_by_filter(VectorFilter(document_id=doc), {"is_active_version": False})
    await qdrant_repo.update_payload_by_filter(VectorFilter(version_id=v2), {"is_active_version": True})

    assert await qdrant_repo.count(VectorFilter(document_id=doc, is_active_version=True)) == 1
    active = await qdrant_repo.search([1.0, 0.0], top_k=1, filter_=VectorFilter(is_active_version=True))
    assert active[0].content == "v2 text"


async def test_delete_by_filter_refuses_empty_filter(qdrant_repo) -> None:
    await qdrant_repo.ensure_collection(2)
    with pytest.raises(ValueError):
        await qdrant_repo.delete_by_filter(VectorFilter())