"""DOCX parser built on python-docx.

Heading levels come from paragraph styles ("Heading 1" → level 1).
Known limitation (documented, decision D-026): DOCX has no page layout
without a rendering engine, so page numbers are None and citations for
DOCX rely on section paths instead of pages.
"""

from __future__ import annotations

import io

from docx import Document as DocxDocument

from app.ingestion.domain import (
    BlockType,
    ExtractedMetadata,
    ParserOutput,
    StructuralBlock,
)
from app.ingestion.parsers.base import Parser
from app.models.enums import DocumentSourceType


class DocxParser(Parser):
    source_type = DocumentSourceType.DOCX
    name = "docx"

    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput:
        document = DocxDocument(io.BytesIO(content))
        blocks: list[StructuralBlock] = []
        title: str | None = None

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            if style.startswith("heading"):
                level = self._level_from_style(style)
                if title is None and level == 1:
                    title = text
                blocks.append(
                    StructuralBlock(
                        text=text, block_type=BlockType.HEADING, heading_level=level
                    )
                )
            elif style.startswith("list"):
                blocks.append(StructuralBlock(text=text, block_type=BlockType.LIST_ITEM))
            else:
                blocks.append(StructuralBlock(text=text))

        core = document.core_properties
        return ParserOutput(
            blocks=blocks,
            metadata=ExtractedMetadata(
                title=title or (core.title or None),
                author=core.author or None,
                page_count=None,  # layout-free format — see module docstring
            ),
        )

    @staticmethod
    def _level_from_style(style: str) -> int:
        digits = "".join(ch for ch in style if ch.isdigit())
        level = int(digits) if digits else 1
        return max(1, min(level, 6))