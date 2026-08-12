#!/usr/bin/env python3
"""Dev CLI: ingest a local file end-to-end without the API.

Usage:
    python scripts/ingest_file.py path/to/document.pdf

Creates (or reuses) a local 'system' user, submits the file through the
real IngestionService, then runs the job through the real runner.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import LocalFileStore
from app.models import Base
from app.repositories.database import Database
from app.repositories.user import UserRepository
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService

logger = get_logger("scripts.ingest_file")

SYSTEM_USER_EMAIL = "system@local"


async def main(path: Path) -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    database = Database(settings.database)
    await database.initialize()

    # Dev convenience: create schema if missing (migrations are the real path).
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    file_store = LocalFileStore(Path("data/uploads"))
    chunking_registry = load_chunking_registry(settings.embedding.registry_path.replace(
        "embeddings.yaml", "chunking.yaml"))
    pipeline = IngestionPipeline(build_default_registry(), chunking_registry, settings)
    service = IngestionService(file_store, chunking_registry, settings)
    runner = IngestionJobRunner(database, pipeline, file_store, settings)

    content = path.read_bytes()

    async with database.session() as session:
        users = UserRepository(session)
        user = await users.get_by_email(SYSTEM_USER_EMAIL)
        if user is None:
            user = await users.create(email=SYSTEM_USER_EMAIL, hashed_password="unused")
        job = await service.submit_file(
            session=session, owner_id=user.id, filename=path.name, content=content
        )
        await session.commit()
        print(f"submitted job_id={job.id} document_id={job.document_id}")

    processed = await runner.run_next()
    print(f"processed={processed}")

    async with database.session() as session:
        from app.repositories.document import DocumentVersionRepository

        versions = DocumentVersionRepository(session)
        active = await versions.get_active_for_document(job.document_id)
        print(
            f"version={active.version_number if active else None} "
            f"chunks={active.chunk_count if active else None}"
        )

    await database.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.file))