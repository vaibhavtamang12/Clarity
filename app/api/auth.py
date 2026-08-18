"""Authentication abstraction (Phase 16).

ApiKeyAuthProvider is the minimum production mechanism; the AuthProvider
protocol is the seam for JWT/OIDC later — endpoints never change (Phase 2
plan realized). Keys are stored hashed (Phase 3); plaintext exists only at
creation time (scripts/create_api_key.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UnauthorizedError
from app.models.user import User
from app.repositories.user import ApiKeyRepository
from app.utils.security import hash_api_key

BEARER_PREFIX = "Bearer "


class AuthenticatedUser(BaseModel):
    id: uuid.UUID
    email: str
    role: str


def parse_bearer_header(header: str | None) -> str:
    if not header:
        raise UnauthorizedError("Missing Authorization header")
    if not header.startswith(BEARER_PREFIX):
        raise UnauthorizedError("Authorization scheme must be Bearer")
    token = header[len(BEARER_PREFIX) :].strip()
    if not token:
        raise UnauthorizedError("Empty bearer token")
    return token


@runtime_checkable
class AuthProvider(Protocol):
    async def authenticate(self, session: AsyncSession, token: str) -> AuthenticatedUser: ...


class ApiKeyAuthProvider:
    """Hashed API-key lookup with expiry + active checks."""

    async def authenticate(self, session: AsyncSession, token: str) -> AuthenticatedUser:
        repo = ApiKeyRepository(session)
        api_key = await repo.get_by_key_hash(hash_api_key(token))
        if api_key is None or not api_key.is_active:
            raise UnauthorizedError("Invalid API key")
        if api_key.expires_at is not None and api_key.expires_at < datetime.now(timezone.utc):
            raise UnauthorizedError("API key expired")

        user = await session.get(User, api_key.user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("User not found or inactive")

        await repo.touch_last_used(api_key)
        await session.commit()
        role = user.role.value if hasattr(user.role, "value") else str(user.role)
        return AuthenticatedUser(id=user.id, email=user.email, role=role)