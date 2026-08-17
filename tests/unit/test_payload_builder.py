"""Payload assembly from system-of-record rows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models.document import Document, DocumentChunk, DocumentVersion
from app.models.enums import DocumentSourceType, VersionStatus
from app.services.indexing_service import build_payload


def _rows():
    owner_id = uuid.uuid4()
    document = Document(
        id=uuid.uuid4(),
        owner_id=owner_id,
        source_type=DocumentSourceType.PDF,
        title="Refund Policy",
        source_uri="refund_policy.pdf",
        metadata_={"tags": ["legal", "billing"], "department": "legal"},
    )
    version = DocumentVersion(
        id=uuid.uuid4(), document_id=document.id, version_number=1,
        content_hash="h", status=VersionStatus.ACTIVE,
    )
    chunk = DocumentChunk(
        id=uuid.uuid4(), document_id=document.id, version_id=version.id,
        chunk_index=0, content="Enterprise refunds within 30 days.",
        token_count=6, page_start=12, page_end=12, section="Enterprise Refunds",
    )
    chunk.created_at = datetime.now(timezone.utc)
    return document, version, chunk


def test_payload_carries_citation_and_tenant_fields() -> None:
    document, version, chunk = _rows()
    payload = build_payload(chunk=chunk, document=document, version=version, is_active=True)

    assert payload.chunk_id == chunk.id
    assert payload.owner_id == document.owner_id          # tenant isolation field
    assert payload.is_active_version is True
    assert payload.source_type == "pdf"
    assert payload.page_start == 12 and payload.page_end == 12
    assert payload.section == "Enterprise Refunds"
    assert payload.tags == ("legal", "billing")
    assert payload.department == "legal"
    assert payload.document_type == "pdf"                 # falls back to source_type


def test_payload_dict_is_json_safe() -> None:
    document, version, chunk = _rows()
    data = build_payload(chunk=chunk, document=document, version=version, is_active=False).to_dict()
    assert data["document_id"] == str(document.id)
    assert isinstance(data["tags"], list)
    assert isinstance(data["created_at"], str)
    assert data["is_active_version"] is False