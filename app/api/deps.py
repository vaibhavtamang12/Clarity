"""FastAPI dependency providers.

get_db yields a session scoped to the request. Commit/rollback is the
caller's responsibility (unit of work) — the session closes either way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.database import Database


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.database
    async with database.session() as session:
        yield session