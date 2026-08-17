"""Qdrant client factory — the ONLY place an AsyncQdrantClient is created.

Construction is lazy-connect: building the client opens no sockets, so app
startup works even while Qdrant is still booting (same posture as Database).
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.core.config import QdrantSettings


def build_qdrant_client(settings: QdrantSettings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.host,
        port=settings.port,
        grpc_port=settings.grpc_port,
        prefer_grpc=settings.prefer_grpc,
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
        timeout=settings.timeout_seconds,
    )