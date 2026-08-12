"""Integration fixtures — real PostgreSQL, isolated test database.

Provisioning: connects to the `postgres` maintenance DB and creates
`rag_platform_test` if missing. If PostgreSQL is unreachable (e.g. local
dev without Docker), tests skip gracefully instead of failing — CI runs
them against a service container (Phase 28).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models import Base
from app.repositories.database import Database

TEST_DB_NAME = "rag_platform_test"


async def _ensure_test_database(base_url: str) -> None:
    """Create the test database via the maintenance DB if it doesn't exist."""
    admin_url = base_url.rsplit("/", 1)[0] + "/postgres"
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            exists = (
                await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": TEST_DB_NAME},
                )
            ).scalar()
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    finally:
        await admin_engine.dispose()


@pytest_asyncio.fixture()
async def database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Database]:
    monkeypatch.setenv("DATABASE__NAME", TEST_DB_NAME)
    get_settings.cache_clear()
    settings = get_settings()

    try:
        await _ensure_test_database(settings.database.async_url)
    except Exception as exc:  # noqa: BLE001 — provisioning failure = skip
        pytest.skip(f"PostgreSQL not reachable for integration tests: {exc}")

    db = Database(settings.database)
    await db.initialize()
    try:
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # noqa: BLE001
        await db.dispose()
        pytest.skip(f"Could not prepare test schema: {exc}")

    yield db
    await db.dispose()
    get_settings.cache_clear()