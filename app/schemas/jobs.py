# app/schemas/jobs.py
"""Ingestion job API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class IngestionJobResponse(BaseModel):
    job_id: uuid.UUID
    document_id: uuid.UUID
    job_type: str
    status: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    document_id: uuid.UUID
    job_type: str
    status: str
    attempt_count: int
    max_attempts: int
    error_type: str | None = None
    error_message: str | None = None
    progress: dict[str, Any] = {}
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None