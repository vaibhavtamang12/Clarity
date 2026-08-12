"""Generic repository base.

Repositories are the ONLY layer allowed to touch SQLAlchemy/Qdrant/Redis
clients (enforced by import-linter contracts from Phase 2).
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: ClassVar[type[Any]]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        await self.session.flush()
        return instance