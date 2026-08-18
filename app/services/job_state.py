"""Temporary job state in Redis (Phase 18, decision D-102).

Redis holds HOT progress data for fast polling; PostgreSQL remains the
system of record (durable status, attempts, errors). The jobs endpoint
reads Redis first and falls back to the DB — a Redis outage changes
nothing except polling freshness.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from redis.asyncio import Redis

from app.core.logging import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "job:state"
TTL_SECONDS = 7 * 24 * 3600  # 7 days


@runtime_checkable
class JobStateStore(Protocol):
    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None: ...

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None: ...


class RedisJobStateStore:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None:
        try:
            key = f"{KEY_PREFIX}:{job_id}"
            mapping = {k: str(v) for k, v in progress.items()}
            await self._client.hset(key, mapping=mapping)
            await self._client.expire(key, TTL_SECONDS)
        except Exception as exc:  # noqa: BLE001 — fail-open; DB is the truth
            logger.warning("job_state_write_failed", error=type(exc).__name__)

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None:
        try:
            raw = await self._client.hgetall(f"{KEY_PREFIX}:{job_id}")
            if not raw:
                return None
            return {
                k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()
            }
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.warning("job_state_read_failed", error=type(exc).__name__)
            return None


class InMemoryJobStateStore:
    def __init__(self) -> None:
        self._data: dict[uuid.UUID, dict[str, Any]] = {}

    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None:
        self._data[job_id] = dict(progress)

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None:
        return self._data.get(job_id)