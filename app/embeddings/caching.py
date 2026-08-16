"""Embedding cache abstractions.

Cache keys embed model_key AND model_version (decision D-035): after a model
change, stale vectors are unreachable by construction — no invalidation race
exists. Redis is the production store; InMemoryCacheStore serves tests and
offline dev. Caching fails OPEN (decision D-037): a store outage costs
latency, never correctness or availability.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.ingestion.domain import sha256_hex

KEY_PREFIX = "cache:emb"


@runtime_checkable
class CacheStore(Protocol):
    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...


class InMemoryCacheStore:
    """Dict-backed store for tests and no-Redis development."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self.set_calls = 0  # observable in tests

    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        return {key: self._data[key] for key in keys if key in self._data}

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self.set_calls += 1
        self._data[key] = value


def embedding_cache_key(model_key: str, model_version: str, text: str) -> str:
    return f"{KEY_PREFIX}:{model_key}:{model_version}:{sha256_hex(text)}"


def serialize_vector(vector: list[float]) -> bytes:
    return json.dumps(vector).encode("utf-8")


def deserialize_vector(raw: bytes) -> list[float]:
    return [float(x) for x in json.loads(raw.decode("utf-8"))]