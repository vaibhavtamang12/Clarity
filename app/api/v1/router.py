"""API v1 router — single aggregation point for all endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import chat, conversations, documents, health, jobs, metrics, search

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(search.router)
api_v1_router.include_router(conversations.router)
api_v1_router.include_router(jobs.router)
api_v1_router.include_router(metrics.router)