"""Async engine + session management.

- The engine connects lazily: creating a Database never opens a connection,
  so app startup (and tests) work even before PostgreSQL is reachable.
- pool_pre_ping=True: dead connections are recycled transparently —
  standard defense against idle-timeout disconnects in production.
- Sessions follow the unit-of-work pattern: repositories flush, the caller
  (service layer / dependency) commits or rolls back.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import DatabaseSettings
from app.schemas.health import ComponentHealth

HEALTH_CHECK_TIMEOUT_SECONDS = 3.0


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        return self._engine

    async def initialize(self) -> None:
        """Create engine + session factory. Idempotent. No connection is opened."""
        if self._engine is not None:
            return
        self._engine = create_async_engine(
            self._settings.async_url,
            pool_size=self._settings.pool_size,
            max_overflow=self._settings.max_overflow,
            pool_pre_ping=True,
            echo=self._settings.echo,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        async with self._session_factory() as session:
            yield session

    async def health_check(self) -> ComponentHealth:
        """Probe with SELECT 1 under a hard timeout. Never raises."""
        start = time.perf_counter()
        try:
            async with asyncio.timeout(HEALTH_CHECK_TIMEOUT_SECONDS):
                async with self.session() as session:
                    await session.execute(text("SELECT 1"))
            latency_ms = (time.perf_counter() - start) * 1000
            return ComponentHealth(status="healthy", latency_ms=round(latency_ms, 2))
        except Exception as exc:  # noqa: BLE001 — health checks must not raise
            # Exception *class name* only — never leak driver details (Rule 10).
            return ComponentHealth(status="unavailable", detail=type(exc).__name__)