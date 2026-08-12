"""Integration tests: models, constraints, and repositories against real PostgreSQL.

Run with `make test-integration` (requires PostgreSQL — `docker compose up -d postgres`).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
    IngestionJob,
    User,
)
from app.models.enums import DocumentSourceType, DocumentStatus, JobStatus, VersionStatus
from app.repositories.database import Database
from app.repositories.document import (
    DocumentChunkRepository,
    DocumentRepository,
    DocumentVersionRepository,
)
from app.repositories.job import IngestionJobRepository
from app.repositories.user import UserRepository

pytestmark = pytest.mark.integration


async def _make_user(db: Database, email: str = "owner@example.com") -> User:
    async with db.session() as session:
        user = await UserRepository(session).create(
            email=email, hashed_password="hashed-not-real"
        )
        await session.commit()
        return user


async def test_user_roundtrip(database: Database) -> None:
    async with database.session() as session:
        repo = UserRepository(session)
        await repo.create(email="alice@example.com", hashed_password="x", full_name="Alice")
        await session.commit()

    async with database.session() as session:
        found = await UserRepository(session).get_by_email("alice@example.com")
        assert found is not None
        assert found.full_name == "Alice"
        assert found.is_active is True


async def test_version_activation_supersedes_previous(database: Database) -> None:
    user = await _make_user(database)
    async with database.session() as session:
        docs, versions = DocumentRepository(session), DocumentVersionRepository(session)
        doc = await docs.create(owner_id=user.id, source_type=DocumentSourceType.PDF, title="Policy")
        v1 = await versions.create(doc.id, 1, content_hash="hash-v1")
        v2 = await versions.create(doc.id, 2, content_hash="hash-v2")
        await versions.activate(v1.id)
        await session.commit()

        activated = await versions.activate(v2.id)
        await session.commit()

        assert activated.status == VersionStatus.ACTIVE
        active = await versions.get_active_for_document(doc.id)
        assert active is not None and active.id == v2.id
        refreshed_v1 = await versions.get_by_id(v1.id)
        assert refreshed_v1 is not None and refreshed_v1.status == VersionStatus.SUPERSEDED


async def test_partial_unique_index_blocks_two_active_versions(database: Database) -> None:
    user = await _make_user(database, email="constraint@example.com")
    async with database.session() as session:
        docs, versions = DocumentRepository(session), DocumentVersionRepository(session)
        doc = await docs.create(owner_id=user.id, source_type=DocumentSourceType.TXT)
        v1 = await versions.create(doc.id, 1, content_hash="h1")
        await versions.activate(v1.id)
        v2 = await versions.create(doc.id, 2, content_hash="h2")
        v2.status = VersionStatus.ACTIVE  # bypass the handshake on purpose
        with pytest.raises(IntegrityError):
            await session.flush()


async def test_chunk_unique_version_and_index(database: Database) -> None:
    user = await _make_user(database, email="chunks@example.com")
    async with database.session() as session:
        docs, versions, chunks = (
            DocumentRepository(session),
            DocumentVersionRepository(session),
            DocumentChunkRepository(session),
        )
        doc = await docs.create(owner_id=user.id, source_type=DocumentSourceType.MARKDOWN)
        version = await versions.create(doc.id, 1, content_hash="h")
        created = await chunks.bulk_create(
            [
                DocumentChunk(
                    document_id=doc.id, version_id=version.id, chunk_index=0, content="Alpha"
                ),
                DocumentChunk(
                    document_id=doc.id, version_id=version.id, chunk_index=1, content="Beta"
                ),
            ]
        )
        assert created == 2
        assert await chunks.count_for_version(version.id) == 2

        # Duplicate (version_id, chunk_index) must violate the unique constraint.
        session.add(
            DocumentChunk(document_id=doc.id, version_id=version.id, chunk_index=0, content="Dup")
        )
        with pytest.raises(IntegrityError):
            await session.flush()


async def test_job_idempotency_and_claim(database: Database) -> None:
    user = await _make_user(database, email="jobs@example.com")
    async with database.session() as session:
        docs, jobs = DocumentRepository(session), IngestionJobRepository(session)
        doc = await docs.create(owner_id=user.id, source_type=DocumentSourceType.PDF)
        job = await jobs.create(doc.id, idempotency_key="key-abc")
        await jobs.enqueue(job)
        await session.commit()

        assert (await jobs.get_by_idempotency_key("key-abc")) is not None

        # Duplicate idempotency key must be rejected at the DB level.
        session.add(
            IngestionJob(document_id=doc.id, idempotency_key="key-abc")
        )
        with pytest.raises(IntegrityError):
            await session.flush()

    async with database.session() as session:
        jobs = IngestionJobRepository(session)
        claimed = await jobs.claim_next()
        await session.commit()
        assert claimed is not None
        assert claimed.status == JobStatus.PROCESSING
        assert claimed.started_at is not None

        # Queue is now empty.
        assert await jobs.claim_next() is None