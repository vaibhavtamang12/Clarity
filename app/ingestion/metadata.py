"""Metadata extraction: parser-provided metadata + content-derived fallbacks."""

from __future__ import annotations

from app.ingestion.domain import ExtractedMetadata, ParserOutput

_MAX_TITLE_CHARS = 200


def extract_metadata(output: ParserOutput, fallback_title: str | None = None) -> ExtractedMetadata:
    title = output.metadata.title
    if not title:
        first_heading = next((b.text for b in output.blocks if b.is_heading), None)
        title = first_heading or fallback_title
    if title:
        title = title.strip()[:_MAX_TITLE_CHARS]
    return ExtractedMetadata(
        title=title,
        author=output.metadata.author,
        page_count=output.metadata.page_count,
        extra=output.metadata.extra,
    )