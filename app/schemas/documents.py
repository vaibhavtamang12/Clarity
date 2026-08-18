# app/schemas/documents.py
"""Document/version API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    source_type: str
    source_uri: str | None
    status: str
    content_hash: str | None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    limit: int
    offset: int
    has_more: bool


class VersionResponse(BaseModel):
    version_number: int
    status: str
    content_hash: str
    chunk_count: int | None
    token_count: int | None
    page_count: int | None
    embedding_model: str | None
    parser: str | None
    created_at: datetime


class VersionListResponse(BaseModel):
    items: list[VersionResponse]


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)