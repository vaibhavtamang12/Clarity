"""RetrievalLog + EvaluationRun repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select

from app.models.logs import EvaluationRun, RetrievalLog
from app.repositories.base import BaseRepository


class RetrievalLogRepository(BaseRepository[RetrievalLog]):
    model = RetrievalLog

    async def create(self, **fields: object) -> RetrievalLog:
        return await self.add(RetrievalLog(**fields))  # type: ignore[arg-type]


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
    model = EvaluationRun

    async def create(self, **fields: object) -> EvaluationRun:
        return await self.add(EvaluationRun(**fields))  # type: ignore[arg-type]

    async def list_recent(self, limit: int = 20) -> Sequence[EvaluationRun]:
        result = await self.session.execute(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        )
        return result.scalars().all()