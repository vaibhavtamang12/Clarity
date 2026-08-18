# app/services/index_state.py — FULL UPDATED FILE
"""Index state notification seam (decision D-081, Redis realized D-098).

Every index mutation bumps a monotonic counter scoped to the collection.
Response cache keys incorporate the counter, so stale-cache-after-version-
change is impossible by construction — keys change; nothing is deleted.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from redis.asyncio import Redis


@runtime_checkable
class IndexStateNotifier(Protocol):
    async def bump(self, scope: str) -> int:
        """Advance the state counter for a scope; returns the new value."""
        ...

    async def current(self, scope: str) -> int:
        """Current counter value (0 if never bumped)."""
        ...


class LocalIndexStateNotifier:
    """In-process counter: single-replica deployments, tests, Redis-down fallback."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    async def bump(self, scope: str) -> int:
        self._counters[scope] = self._counters.get(scope, 0) + 1
        return self._counters[scope]

    async def current(self, scope: str) -> int:
        return self._counters.get(scope, 0)


class RedisIndexStateNotifier:
    """Production counter: INCR index:state:{scope}."""

    KEY_PREFIX = "index:state"

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def bump(self, scope: str) -> int:
        return int(await self._client.incr(f"{self.KEY_PREFIX}:{scope}"))

    async def current(self, scope: str) -> int:
        raw = await self._client.get(f"{self.KEY_PREFIX}:{scope}")
        return int(raw) if raw is not None else 0