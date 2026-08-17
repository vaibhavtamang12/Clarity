"""Qdrant implementation of the VectorRepository contract.

Design points:
- Vendor types never leak past this module; callers see VectorFilter /
  SearchResult only (ADR-001, import-linter contract).
- build_qdrant_filter is a PURE function → unit-testable without a server.
- Every network call goes through _execute: bounded retries, typed failure
  (VectorStoreUnavailableError), structured logs (Rule 10).
- Payload indexes are created once per collection for every filterable
  field, so filtered search stays fast as data grows.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from app.core.exceptions import VectorStoreUnavailableError
from app.core.logging import get_logger
from app.repositories.vector.base import (
    SearchResult,
    VectorFilter,
    VectorPoint,
)

logger = get_logger(__name__)

MAX_ATTEMPTS = 2
RETRY_BASE_DELAY_SECONDS = 0.2
UPSERT_BATCH_SIZE = 256

# field name → qdrant payload index schema (ARCHITECTURE.md §5.2)
_PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("document_id", "keyword"),
    ("version_id", "keyword"),
    ("owner_id", "keyword"),
    ("source_type", "keyword"),
    ("document_type", "keyword"),
    ("department", "keyword"),
    ("tags", "keyword"),
    ("is_active_version", "bool"),
    ("created_at", "datetime"),
    ("page_start", "integer"),
)


def build_qdrant_filter(filter_: VectorFilter | None) -> qmodels.Filter | None:
    """Pure mapping: declarative VectorFilter → Qdrant filter conditions."""
    if filter_ is None or filter_.is_empty():
        return None
    conditions: list[qmodels.Condition] = []
    if filter_.document_id:
        conditions.append(qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=str(filter_.document_id))))
    if filter_.version_id:
        conditions.append(qmodels.FieldCondition(key="version_id", match=qmodels.MatchValue(value=str(filter_.version_id))))
    if filter_.owner_id:
        conditions.append(qmodels.FieldCondition(key="owner_id", match=qmodels.MatchValue(value=str(filter_.owner_id))))
    if filter_.source_type:
        conditions.append(qmodels.FieldCondition(key="source_type", match=qmodels.MatchValue(value=filter_.source_type)))
    if filter_.document_type:
        conditions.append(qmodels.FieldCondition(key="document_type", match=qmodels.MatchValue(value=filter_.document_type)))
    if filter_.department:
        conditions.append(qmodels.FieldCondition(key="department", match=qmodels.MatchValue(value=filter_.department)))
    if filter_.tags:
        conditions.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=list(filter_.tags))))
    if filter_.is_active_version is not None:
        conditions.append(qmodels.FieldCondition(key="is_active_version", match=qmodels.MatchValue(value=filter_.is_active_version)))
    if filter_.created_after or filter_.created_before:
        conditions.append(
            qmodels.FieldCondition(
                key="created_at",
                range=qmodels.Range(
                    gte=filter_.created_after.isoformat() if filter_.created_after else None,
                    lte=filter_.created_before.isoformat() if filter_.created_before else None,
                ),
            )
        )
    return qmodels.Filter(must=conditions)


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class QdrantVectorRepository:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection

    # ---------------------------------------------------------------- lifecycle
    async def ensure_collection(self, dimension: int) -> None:
        exists = await self._execute(
            "collection_exists", lambda: self._client.collection_exists(self.collection)
        )
        if exists:
            return
        await self._execute(
            "create_collection",
            lambda: self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dimension, distance=qmodels.Distance.COSINE),
            ),
        )
        for field_name, schema in _PAYLOAD_INDEXES:
            try:
                await self._execute(
                    f"create_payload_index:{field_name}",
                    lambda f=field_name, s=schema: self._client.create_payload_index(
                        collection_name=self.collection, field_name=f, field_schema=s
                    ),
                )
            except VectorStoreUnavailableError as exc:
                # Index creation is an optimization; don't fail ingestion over it,
                # but make it visible.
                logger.warning("payload_index_creation_failed", field=field_name, error=str(exc))

    async def collection_exists(self) -> bool:
        return bool(await self._execute("collection_exists", lambda: self._client.collection_exists(self.collection)))

    async def delete_collection(self) -> None:
        await self._execute("delete_collection", lambda: self._client.delete_collection(self.collection))

    # ------------------------------------------------------------------- writes
    async def upsert_points(self, points: Sequence[VectorPoint]) -> int:
        total = 0
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            structs = [
                qmodels.PointStruct(
                    id=str(point.point_id),
                    vector=point.vector,
                    payload=point.payload.to_dict(),
                )
                for point in batch
            ]
            await self._execute(
                f"upsert[{len(structs)}]",
                lambda s=structs: self._client.upsert(collection_name=self.collection, points=s, wait=True),
            )
            total += len(structs)
        return total

    async def delete_points(self, point_ids: Sequence[uuid.UUID]) -> None:
        if not point_ids:
            return
        await self._execute(
            "delete_points",
            lambda: self._client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(points=[str(pid) for pid in point_ids]),
            ),
        )

    async def delete_by_filter(self, filter_: VectorFilter) -> None:
        qfilter = build_qdrant_filter(filter_)
        if qfilter is None:
            raise ValueError("delete_by_filter requires a non-empty filter — refusing to wipe the collection")
        await self._execute(
            "delete_by_filter",
            lambda: self._client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(filter=qfilter),
            ),
        )

    async def update_payload_by_filter(self, filter_: VectorFilter, payload: dict[str, Any]) -> None:
        qfilter = build_qdrant_filter(filter_)
        if qfilter is None:
            raise ValueError("update_payload_by_filter requires a non-empty filter")
        await self._execute(
            "set_payload",
            lambda: self._client.set_payload(
                collection_name=self.collection,
                payload=payload,
                points_selector=qmodels.FilterSelector(filter=qfilter),
            ),
        )

    # --------------------------------------------------------------------- reads
    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filter_: VectorFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        response = await self._execute(
            "search",
            lambda: self._client.query_points(
                collection_name=self.collection,
                query=list(query_vector),
                limit=top_k,
                query_filter=build_qdrant_filter(filter_),
                with_payload=True,
                score_threshold=score_threshold,
            ),
        )
        return [
            SearchResult.from_payload(_as_uuid(point.id), float(point.score), point.payload or {})
            for point in response.points
        ]

    async def count(self, filter_: VectorFilter | None = None) -> int:
        result = await self._execute(
            "count",
            lambda: self._client.count(
                collection_name=self.collection,
                count_filter=build_qdrant_filter(filter_),
                exact=True,
            ),
        )
        return int(result.count)

    # ------------------------------------------------------------------ resilience
    async def _execute(self, operation: str, func: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await func()
            except Exception as exc:  # noqa: BLE001 — normalize all vendor/network failures
                last_error = exc
                logger.warning(
                    "qdrant_operation_failed",
                    operation=operation,
                    collection=self.collection,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * attempt)
        raise VectorStoreUnavailableError(
            f"Qdrant operation '{operation}' failed after {MAX_ATTEMPTS} attempts"
        ) from last_error