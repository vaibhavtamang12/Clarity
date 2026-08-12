"""Deterministic identifiers.

Qdrant point IDs are UUIDv5(version_id, chunk_index): the same chunk of the
same version always maps to the same point, which makes vector upserts
idempotent and crash-safe (ARCHITECTURE.md §5.2).
"""

from __future__ import annotations

import uuid

QDRANT_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-platform/qdrant-point")


def chunk_point_id(version_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(QDRANT_POINT_NAMESPACE, f"{version_id}:{chunk_index}")