"""Centralized domain enums shared by ORM models, schemas, and services.

All enums are StrEnums so their *values* (not member names) are what get
serialized and stored — see the ``StrEnumColumn`` helper in base.py.
"""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class DocumentSourceType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class VersionStatus(StrEnum):
    PROCESSING = "processing"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    ARCHIVED = "archived"


class JobType(StrEnum):
    INGEST = "ingest"
    REINDEX = "reindex"
    DELETE_INDEX = "delete_index"


class JobStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EvaluationRunType(StrEnum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    END_TO_END = "end_to_end"