"""Health endpoints.

/health — readiness-style view with per-dependency checks.
PostgreSQL: SELECT 1 probe. Qdrant: get_collections probe. Redis probe lands
with Phase 18.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.config import Settings
from app.repositories.database import Database
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])

QDRANT_PROBE_TIMEOUT_SECONDS = 3.0


def _not_configured() -> ComponentHealth:
    return ComponentHealth(status="not_configured", detail="probe wired in a later phase")


async def _probe_qdrant(client: object) -> ComponentHealth:
    start = time.perf_counter()
    try:
        async with asyncio.timeout(QDRANT_PROBE_TIMEOUT_SECONDS):
            await client.get_collections()  # type: ignore[union-attr]
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(status="healthy", latency_ms=round(latency_ms, 2))
    except Exception as exc:  # noqa: BLE001 — health probes never raise
        return ComponentHealth(status="unavailable", detail=type(exc).__name__)

async def _probe_redis(client: object) -> ComponentHealth:
    start = time.perf_counter()
    try:
        async with asyncio.timeout(3.0):
            await client.ping()  # type: ignore[union-attr]
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(status="healthy", latency_ms=round(latency_ms, 2))
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status="unavailable", detail=type(exc).__name__)

# In health():
    redis_client = getattr(request.app.state, "redis_client", None)
    redis_check = await _probe_redis(redis_client) if redis_client is not None else _not_configured()
    checks = {
        "postgres": postgres_check,
        "redis": redis_check,
        "qdrant": qdrant_check,
    }

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    database: Database | None = getattr(request.app.state, "database", None)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)

    postgres_check = await database.health_check() if database is not None else _not_configured()
    qdrant_check = await _probe_qdrant(qdrant_client) if qdrant_client is not None else _not_configured()

    checks = {
        "postgres": postgres_check,
        "redis": _not_configured(),
        "qdrant": qdrant_check,
    }
    return HealthResponse(
        status="ok",
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment.value,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )