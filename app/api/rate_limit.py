"""Rate limiting (Phase 18, decision D-099).

Sliding-window log over a Redis sorted set:
- ZREMRANGEBYSCORE drops entries outside the 60s window
- ZADD records this request
- ZCARD counts the window
All in one atomic pipeline. Denied requests are removed again so retries
aren't penalized twice.

Fail-open per ARCHITECTURE §9: a Redis outage degrades rate limiting with a
structured warning — it never takes the API down. The in-memory limiter is
the test/single-replica fallback with identical semantics.
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fastapi import Depends, Request
from redis.asyncio import Redis

from app.core.exceptions import RateLimitedError
from app.core.logging import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS = 60.0
KEY_PREFIX = "rl"


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: float


@runtime_checkable
class RateLimiter(Protocol):
    async def check(self, identity: str, limit_per_minute: int) -> RateLimitResult: ...


class RedisRateLimiter:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def check(self, identity: str, limit_per_minute: int) -> RateLimitResult:
        key = f"{KEY_PREFIX}:{identity}"
        now = time.time()
        window_start = now - WINDOW_SECONDS
        member = f"{now}:{secrets.token_hex(4)}"
        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)
            pipe.expire(key, int(WINDOW_SECONDS) + 1)
            results = await pipe.execute()
        except Exception as exc:  # noqa: BLE001 — fail-open (ARCHITECTURE §9)
            logger.warning("rate_limiter_redis_failed_open", error=type(exc).__name__)
            return RateLimitResult(allowed=True, remaining=limit_per_minute, retry_after_seconds=0.0)

        count = int(results[2])
        if count <= limit_per_minute:
            return RateLimitResult(
                allowed=True,
                remaining=limit_per_minute - count,
                retry_after_seconds=0.0,
            )

        # Denied: remove the recorded attempt so retries aren't double-counted.
        try:
            await self._client.zrem(key, member)
        except Exception:  # noqa: BLE001
            pass
        oldest_entries = results[3]
        oldest = float(oldest_entries[0][1]) if oldest_entries else now
        retry_after = max(0.0, (oldest + WINDOW_SECONDS) - now)
        return RateLimitResult(allowed=False, remaining=0, retry_after_seconds=round(retry_after, 2))


class InMemoryRateLimiter:
    """Same sliding-window semantics, process-local (tests / Redis-down)."""

    def __init__(self, now_fn: Callable[[], float] = time.time) -> None:
        self._now = now_fn
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def check(self, identity: str, limit_per_minute: int) -> RateLimitResult:
        now = self._now()
        window_start = now - WINDOW_SECONDS
        hits = self._hits[identity]
        while hits and hits[0] <= window_start:
            hits.popleft()
        if len(hits) < limit_per_minute:
            hits.append(now)
            return RateLimitResult(
                allowed=True, remaining=limit_per_minute - len(hits), retry_after_seconds=0.0
            )
        retry_after = max(0.0, (hits[0] + WINDOW_SECONDS) - now)
        return RateLimitResult(allowed=False, remaining=0, retry_after_seconds=round(retry_after, 2))


async def rate_limit_dependency(
    request: Request,
    user=Depends(__import__("app.api.deps", fromlist=["get_current_user"]).get_current_user),
) -> None:
    """Enforce per-user rate limits on authenticated endpoints."""
    from app.api.deps import get_platform

    platform = get_platform(request)
    result = await platform.rate_limiter.check(
        f"user:{user.id}", platform.settings.security.rate_limit_per_minute
    )
    counters = getattr(request.app.state, "metrics", None)
    if not result.allowed:
        if counters is not None:
            counters["rate_limit_rejected_total"] = (
                counters.get("rate_limit_rejected_total", 0) + 1
            )
        raise RateLimitedError(
            "Rate limit exceeded",
            details={"retry_after_seconds": result.retry_after_seconds},
        )
    request.state.rate_limit_remaining = result.remaining