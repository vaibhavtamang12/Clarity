"""Unit tests for deterministic point IDs and idempotency keys."""

from __future__ import annotations

import uuid

from app.ingestion.idempotency import compute_idempotency_key
from app.utils.ids import chunk_point_id


def test_point_ids_deterministic_and_unique() -> None:
    version_id = uuid.uuid4()
    a1 = chunk_point_id(version_id, 0)
    a2 = chunk_point_id(version_id, 0)
    b = chunk_point_id(version_id, 1)
    assert a1 == a2
    assert a1 != b
    assert chunk_point_id(uuid.uuid4(), 0) != a1


def test_idempotency_key_stable_and_sensitive() -> None:
    kwargs = {
        "content_hash": "abc",
        "parser": "pdf",
        "chunking_strategy": "structure_aware_512",
        "chunking_config": {"target_tokens": 512},
        "embedding_model": "bge_m3",
    }
    assert compute_idempotency_key(**kwargs) == compute_idempotency_key(**kwargs)
    changed = {**kwargs, "content_hash": "abd"}
    assert compute_idempotency_key(**changed) != compute_idempotency_key(**kwargs)