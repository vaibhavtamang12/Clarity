"""In-memory vector repository with real cosine semantics.

Purpose (decision D-039):
- Unit-test every repository semantic (filters, idempotent upserts, payload
  updates, active-version flips) with zero infrastructure.
- Offline development mode with identical behavior to Qdrant.

Not for production: no persistence, no ANN — exact brute-force scoring,
which is precisely why test results are trustworthy.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.repositories.vector.base import (
    SearchResult,
    VectorFilter,
    VectorPayload,
    VectorPoint,
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorRepository:
    def __init__(self, collection: str = "in_memory") -> None:
        self.collection = collection
        self._dimension: int | None = None
        self._points: dict[uuid.UUID, tuple[list[float], dict[str, Any]]] = {}

    # ---------------------------------------------------------------- lifecycle
    async def ensure_collection(self, dimension: int) -> None:
        if self._dimension is not None and self._dimension != dimension:
            raise ValueError(
                f"Collection '{self.collection}' already has dimension {self._dimension}"
            )
        self._dimension = dimension

    async def collection_exists(self) -> bool:
        return self._dimension is not None

    async def delete_collection(self) -> None:
        self._dimension = None
        self._points.clear()

    # ------------------------------------------------------------------- writes
    async def upsert_points(self, points: Sequence[VectorPoint]) -> int:
        await self.ensure_collection(len(points[0].vector) if points else 0)
        for point in points:
            if self._dimension and len(point.vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension {len(point.vector)} != collection {self._dimension}"
                )
            self._points[point.point_id] = (list(point.vector), point.payload.to_dict())
        return len(points)

    async def delete_points(self, point_ids: Sequence[uuid.UUID]) -> None:
        for point_id in point_ids:
            self._points.pop(point_id, None)

    async def delete_by_filter(self, filter_: VectorFilter) -> None:
        doomed = [pid for pid, (_, payload) in self._points.items() if _matches(payload, filter_)]
        for point_id in doomed:
            del self._points[point_id]

    async def update_payload_by_filter(self, filter_: VectorFilter, payload: dict[str, Any]) -> None:
        for _pid, (vector, existing) in self._points.items():
            if _matches(existing, filter_):
                existing.update(payload)

    # --------------------------------------------------------------------- reads
    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filter_: VectorFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        filter_ = filter_ or VectorFilter()
        scored: list[tuple[float, uuid.UUID, dict[str, Any]]] = []
        for point_id, (vector, payload) in self._points.items():
            if not _matches(payload, filter_):
                continue
            score = _cosine(query_vector, vector)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append((score, point_id, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult.from_payload(point_id, score, payload)
            for score, point_id, payload in scored[:top_k]
        ]

    async def count(self, filter_: VectorFilter | None = None) -> int:
        filter_ = filter_ or VectorFilter()
        return sum(1 for _, payload in self._points.values() if _matches(payload, filter_))


def _matches(payload: dict[str, Any], filter_: VectorFilter) -> bool:
    if filter_.document_id and payload.get("document_id") != str(filter_.document_id):
        return False
    if filter_.version_id and payload.get("version_id") != str(filter_.version_id):
        return False
    if filter_.owner_id and payload.get("owner_id") != str(filter_.owner_id):
        return False
    if filter_.source_type and payload.get("source_type") != filter_.source_type:
        return False
    if filter_.document_type and payload.get("document_type") != filter_.document_type:
        return False
    if filter_.department and payload.get("department") != filter_.department:
        return False
    if filter_.tags and not set(filter_.tags) & set(payload.get("tags") or []):
        return False
    if filter_.is_active_version is not None and payload.get("is_active_version") != filter_.is_active_version:
        return False
    created_raw = payload.get("created_at")
    if created_raw and (filter_.created_after or filter_.created_before):
        created = datetime.fromisoformat(str(created_raw))
        if filter_.created_after and created < filter_.created_after:
            return False
        if filter_.created_before and created > filter_.created_before:
            return False
    return True