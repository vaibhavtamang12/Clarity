"""Malicious document guards (Phase 25, decision D-139).

Runs at VALIDATION time — before any parser executes — so hostile files are
rejected at the door:

- PDF: page-count cap (parser DoS protection)
- DOCX (zip container):
    * entry-count cap
    * uncompressed-size cap (zip bomb protection)
    * unsafe entry paths (zip slip protection)

A document that fails safety validation can never reach the parsing stage,
the chunker, or the index.
"""

from __future__ import annotations

import io
import zipfile

import pymupdf

from app.core.config import IngestionSettings
from app.core.exceptions import IngestionValidationError
from app.models.enums import DocumentSourceType


def validate_pdf_safety(content: bytes, settings: IngestionSettings) -> None:
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise IngestionValidationError("PDF failed safety validation") from exc
    try:
        if doc.page_count > settings.max_pdf_pages:
            raise IngestionValidationError(
                f"PDF exceeds the {settings.max_pdf_pages}-page limit",
                details={"pages": doc.page_count},
            )
    finally:
        doc.close()


def validate_docx_safety(content: bytes, settings: IngestionSettings) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise IngestionValidationError("DOCX is not a valid zip container") from exc

    with archive:
        entries = archive.infolist()
        if len(entries) > settings.max_zip_entries:
            raise IngestionValidationError(
                f"DOCX exceeds the {settings.max_zip_entries}-entry limit",
                details={"entries": len(entries)},
            )

        total_uncompressed = sum(entry.file_size for entry in entries)
        max_uncompressed_bytes = settings.max_uncompressed_mb * 1024 * 1024
        if total_uncompressed_bytes > max_uncompressed_bytes:
            raise IngestionValidationError(
                "DOCX uncompressed size exceeds the limit (zip bomb protection)",
                details={"uncompressed_bytes": total_uncompressed_bytes},
            )

        for entry in entries:
            parts = entry.filename.split("/")
            if entry.filename.startswith("/") or ".." in parts:
                raise IngestionValidationError(
                    "DOCX contains an unsafe entry path (zip slip protection)"
                )


def validate_binary_safety(
    content: bytes, source_type: DocumentSourceType, settings: IngestionSettings
) -> None:
    if source_type == DocumentSourceType.PDF:
        validate_pdf_safety(content, settings)
    elif source_type == DocumentSourceType.DOCX:
        validate_docx_safety(content, settings)