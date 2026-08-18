"""Index state notification seam (decision D-081).

Every index mutation (new version indexed, rollback, re-index) bumps a
monotonic counter scoped to the collection. Cache keys incorporate this
counter (ARCHITECTURE §5.3: `cache:rag:v1:{index_state}:{query_hash}`),
which makes stale-cache-after-version-change IMPOSSIBLE by construction —
there is no invalidation race to get right.

Phase 15 ships the local implementation; Phase 18 swaps in the Redis
implementation (`INCR index:state:{collection}`) behind the same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexStateNotifier(Protocol):
    def bump(self, scope: str) -> int:
        """Advance the state counter for a scope; returns the new value."""
        ...

    def current(self, scope: str) -> int:
        """Current counter value (0 if never bumped)."""
        ...


class LocalIndexStateNotifier:
    """In-process counter. Adequate for single-API deployments and tests;
    multi-replica deployments get the Redis implementation in Phase 18."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def bump(self, scope: str) -> int:
        self._counters[scope] = self._counters.get(scope, 0) + 1
        return self._counters[scope]

    def current(self, scope: str) -> int:
        return self._counters.get(scope, 0)