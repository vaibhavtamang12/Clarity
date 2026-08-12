"""Health/readiness response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ComponentStatus = Literal["healthy", "degraded", "unavailable", "not_configured"]


class ComponentHealth(BaseModel):
    status: ComponentStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    service: str
    version: str
    environment: str
    timestamp: datetime
    checks: dict[str, ComponentHealth] = Field(default_factory=dict)