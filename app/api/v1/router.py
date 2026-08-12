"""API v1 router — single aggregation point for all endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)