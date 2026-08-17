"""Vector repository abstraction (ADR-001).

The contract the rest of the system programs against. Two implementations:
- QdrantVectorRepository  — production (Qdrant)
- InMemoryVectorRepository — tests / offline dev, identical semantics

Business rules travel as data (VectorFilter / VectorPayload), never as
vendor query objects — that's what keeps Weaviate/others swappable (Rule 4).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VectorFilter:
    """Declarative metadata filter — maps 1:1 to ARCHITECTURE.md §5.2 payloads."""

    document_id: uuid.UUID | None = None
    version_id: uuid.UUID | None = None
    owner_id: uuid.UUID | None = None
    source_type: str | None = None          # "source" filter (pdf/docx/...)
    document_type: str | None = None        # business label from document metadata
    department: str | None = None
    tags: tuple[str, ...] | None = None     # match ANY of the tags
    is_active_version: bool | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    def is_empty(self) -> bool:
        return all(value is None for value in (
            self.document_id, self.version_id, self.owner_id, self.source_type,
            self.document_type, self.department, self.tags, self.is_active_version,
            self.created_after, self.created_before,
        ))


@dataclass(frozen=True)
class VectorPayload:
    """Everything a point carries besides its vector (ARCHITECTURE.md §5.2)."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    owner_id: uuid.UUID
    is_active_version: bool
    source_type: str
    content: str
    token_count: int
    created_at: datetime
    source_uri: str | None = None
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    tags: tuple[str, ...] = ()
    department: str | None = None
    document_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict — what actually gets stored in the vector DB."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "version_id": str(self.version_id),
            "owner_id": str(self.owner_id),
            "is_active_version": self.is_active_version,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "title": self.title,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "tags": list(self.tags),
            "department": self.department,
            "document_type": self.document_type,
            "token_count": self.token_count,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class VectorPoint:
    point_id: uuid.UUID
    vector: list[float]
    payload: VectorPayload


@dataclass(frozen=True)
class SearchResult:
    point_id: uuid.UUID
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    score: float
    content: str
    is_active_version: bool
    source_type: str | None = None
    source_uri: str | None = None
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    @classmethod
    def from_payload(cls, point_id: uuid.UUID, score: float, payload: Mapping[str, Any]) -> "SearchResult":
        return cls(
            point_id=point_id,
            chunk_id=uuid.UUID(str(payload["chunk_id"])),
            document_id=uuid.UUID(str(payload["document_id"])),
            version_id=uuid.UUID(str(payload["version_id"])),
            score=score,
            content=str(payload.get("content", "")),
            is_active_version=bool(payload.get("is_active_version", False)),
            source_type=payload.get("source_type"),
            source_uri=payload.get("source_uri"),
            title=payload.get("title"),
            section=payload.get("section"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
        )


@runtime_checkable
class VectorRepository(Protocol):
    """The data-plane contract for any vector store."""

    collection: str

    async def ensure_collection(self, dimension: int) -> None: ...

    async def collection_exists(self) -> bool: ...

    async def delete_collection(self) -> None: ...

    async def upsert_points(self, points: Sequence[VectorPoint]) -> int: ...

    async def delete_points(self, point_ids: Sequence[uuid.UUID]) -> None: ...

    async def delete_by_filter(self, filter_: VectorFilter) -> None: ...

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filter_: VectorFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]: ...

    async def count(self, filter_: VectorFilter | None = None) -> int: ...

    async def update_payload_by_filter(self, filter_: VectorFilter, payload: dict[str, Any]) -> None: ...