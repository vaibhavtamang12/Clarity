"""Health endpoints.

/health — readiness-style view with per-dependency checks.
PostgreSQL is probed for real (SELECT 1, 3s timeout). Redis and Qdrant
checks are wired as their repositories land in later phases.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.core.config import Settings
from app.repositories.database import Database
from app.schemas.health import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])


def _not_configured() -> ComponentHealth:
    return ComponentHealth(status="not_configured", detail="probe wired in a later phase")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    database: Database | None = getattr(request.app.state, "database", None)

    if database is not None:
        postgres_check = await database.health_check()
    else:
        postgres_check = _not_configured()

    checks = {
        "postgres": postgres_check,
        "redis": _not_configured(),
        "qdrant": _not_configured(),
    }
    return HealthResponse(
        status="ok",
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment.value,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )