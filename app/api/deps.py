"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import AuthenticatedUser, parse_bearer_header
from app.repositories.database import Database


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.database
    async with database.session() as session:
        yield session


def get_platform(request: Request):
    platform = getattr(request.app.state, "platform", None)
    if platform is None:
        raise RuntimeError("Platform not initialized")
    return platform


async def get_current_user(
    request: Request, session: AsyncSession = Depends(get_db)
) -> AuthenticatedUser:
    platform = get_platform(request)
    token = parse_bearer_header(request.headers.get("Authorization"))
    return await platform.auth_provider.authenticate(session, token)