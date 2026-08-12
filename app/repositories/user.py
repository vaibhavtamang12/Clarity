"""User + ApiKey repositories."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from app.models.user import ApiKey, User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    async def create(
        self,
        email: str,
        hashed_password: str,
        full_name: str | None = None,
    ) -> User:
        user = User(email=email, hashed_password=hashed_password, full_name=full_name)
        return await self.add(user)

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


class ApiKeyRepository(BaseRepository[ApiKey]):
    model = ApiKey

    async def create(
        self,
        user_id: uuid.UUID,
        key_hash: str,
        name: str | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKey:
        api_key = ApiKey(user_id=user_id, key_hash=key_hash, name=name, expires_at=expires_at)
        return await self.add(api_key)

    async def get_by_key_hash(self, key_hash: str) -> ApiKey | None:
        result = await self.session.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        return result.scalar_one_or_none()

    async def touch_last_used(self, api_key: ApiKey) -> None:
        api_key.last_used_at = datetime.now(timezone.utc)
        await self.session.flush()