"""Redis-backed cache store (production implementation of CacheStore).

Lives in the repository layer because only repositories may import the
Redis client (import-linter contract). Structurally satisfies the
CacheStore protocol defined in app.embeddings.caching.
"""

from __future__ import annotations

from collections.abc import Sequence

from redis.asyncio import Redis


class RedisCacheStore:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        if not keys:
            return {}
        values = await self._client.mget(list(keys))
        return {key: value for key, value in zip(keys, values) if value is not None}

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        await self._client.set(key, value, ex=ttl_seconds)