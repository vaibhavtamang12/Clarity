"""Document endpoints: upload, URL ingest, list, detail, versions, reindex."""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.core.exceptions import DocumentNotFoundError, IngestionValidationError
from app.models.enums import JobType
from app.repositories.document import DocumentRepository, DocumentVersionRepository
from app.repositories.job import IngestionJobRepository
from app.schemas.documents import (
    DocumentListResponse,
    DocumentResponse,
    UrlIngestRequest,
    VersionListResponse,
    VersionResponse,
)
from app.schemas.jobs import IngestionJobResponse

router = APIRouter(prefix="/documents", tags=["documents"])


def _doc_response(doc) -> DocumentResponse:  # type: ignore[no-untyped-def]
    source_type = doc.source_type.value if hasattr(doc.source_type, "value") else str(doc.source_type)
    status = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
    return DocumentResponse(
        id=doc.id, title=doc.title, source_type=source_type, source_uri=doc.source_uri,
        status=status, content_hash=doc.content_hash,
        created_at=doc.created_at, updated_at=doc.updated_at,
    )


async def _owned_document(session, document_id: uuid.UUID, user: AuthenticatedUser):  # type: ignore[no-untyped-def]
    """Ownership-checked fetch — 404 for foreign resources (D-085)."""
    doc = await DocumentRepository(session).get_by_id(document_id)
    if doc is None or doc.owner_id != user.id:
        raise DocumentNotFoundError(f"Document {document_id} not found")
    return doc


@router.post("/upload", response_model=IngestionJobResponse, status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> IngestionJobResponse:
    platform = get_platform(request)
    max_bytes = platform.settings.ingestion.max_file_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise IngestionValidationError(
            f"File exceeds the {platform.settings.ingestion.max_file_size_mb} MB limit"
        )
    job = await platform.ingestion_service.submit_file(
        session=session,
        owner_id=user.id,
        filename=file.filename or "upload",
        content=content,
    )
    await session.commit()
    return IngestionJobResponse(
        job_id=job.id, document_id=job.document_id,
        job_type=job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        created_at=job.created_at,
    )


@router.post("/url", response_model=IngestionJobResponse, status_code=202)
async def ingest_url(
    payload: UrlIngestRequest,
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> IngestionJobResponse:
    platform = get_platform(request)
    try:
        job = await platform.ingestion_service.submit_url(
            session=session, owner_id=user.id, url=payload.url
        )
    except httpx.HTTPError as exc:
        raise IngestionValidationError(f"URL could not be fetched: {type(exc).__name__}") from exc
    await session.commit()
    return IngestionJobResponse(
        job_id=job.id, document_id=job.document_id,
        job_type=job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        created_at=job.created_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> DocumentListResponse:
    docs = await DocumentRepository(session).list_for_owner(
        user.id, limit=limit + 1, offset=offset
    )
    has_more = len(docs) > limit
    return DocumentListResponse(
        items=[_doc_response(d) for d in docs[:limit]],
        limit=limit, offset=offset, has_more=has_more,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> DocumentResponse:
    doc = await _owned_document(session, document_id, user)
    return _doc_response(doc)


@router.get("/{document_id}/versions", response_model=VersionListResponse)
async def list_versions(
    document_id: uuid.UUID,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> VersionListResponse:
    await _owned_document(session, document_id, user)
    versions = await DocumentVersionRepository(session).list_for_document(document_id)
    return VersionListResponse(
        items=[
            VersionResponse(
                version_number=v.version_number,
                status=v.status.value if hasattr(v.status, "value") else str(v.status),
                content_hash=v.content_hash,
                chunk_count=v.chunk_count,
                token_count=v.token_count,
                page_count=v.page_count,
                embedding_model=v.embedding_model,
                parser=v.parser,
                created_at=v.created_at,
            )
            for v in versions
        ]
    )


@router.post("/{document_id}/reindex", response_model=IngestionJobResponse, status_code=202)
async def reindex_document(
    document_id: uuid.UUID,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> IngestionJobResponse:
    """Queue a REINDEX job — the shared runner re-embeds the active version (D-087)."""
    doc = await _owned_document(session, document_id, user)
    jobs = IngestionJobRepository(session)
    idempotency_key = f"reindex:{doc.id}:{doc.content_hash or 'none'}"
    existing = await jobs.get_by_idempotency_key(idempotency_key)
    if existing is None:
        job = await jobs.create(
            doc.id, idempotency_key=idempotency_key, job_type=JobType.REINDEX.value
        )
        await jobs.enqueue(job)
    else:
        job = existing
    await session.commit()
    return IngestionJobResponse(
        job_id=job.id, document_id=job.document_id,
        job_type=job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        created_at=job.created_at,
    )