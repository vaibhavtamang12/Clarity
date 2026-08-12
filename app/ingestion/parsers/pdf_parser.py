"""PDF parser built on PyMuPDF.

Preserves page numbers (citation requirement FR-2.3) and detects headings
via font metrics: a line whose dominant span is bold or significantly
larger than the document's median body size becomes a heading. No layout
model, no OCR — trade-off documented in RISKS.md (R-001).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from dataclasses import dataclass

import pymupdf

from app.ingestion.domain import (
    BlockType,
    ExtractedMetadata,
    ParserOutput,
    StructuralBlock,
)
from app.ingestion.parsers.base import Parser
from app.models.enums import DocumentSourceType

_BOLD_FLAG = 2 ** 4  # pymupdf span flag bit for bold
_HEADING_SIZE_RATIO = 1.15


@dataclass
class _Line:
    text: str
    size: float
    bold: bool


class PdfParser(Parser):
    source_type = DocumentSourceType.PDF
    name = "pdf"

    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput:
        doc = pymupdf.open(stream=content, filetype="pdf")
        try:
            pages = [self._extract_page(page) for page in doc]
            body_size = self._median_body_size(pages)
            blocks: list[StructuralBlock] = []
            for page_number, lines in enumerate(pages, start=1):
                for line in lines:
                    if not line.text.strip():
                        continue
                    is_heading = self._is_heading(line, body_size)
                    blocks.append(
                        StructuralBlock(
                            text=line.text,
                            block_type=BlockType.HEADING if is_heading else BlockType.PARAGRAPH,
                            heading_level=1 if is_heading else 0,
                            page=page_number,
                        )
                    )

            metadata = doc.metadata or {}
            title = (metadata.get("title") or "").strip() or None
            author = (metadata.get("author") or "").strip() or None
            return ParserOutput(
                blocks=blocks,
                metadata=ExtractedMetadata(title=title, author=author, page_count=len(pages)),
            )
        finally:
            doc.close()

    def _extract_page(self, page: "pymupdf.Page") -> list[_Line]:
        lines: list[_Line] = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # text blocks only
                continue
            for raw_line in block.get("lines", []):
                spans = sorted(raw_line.get("spans", []), key=lambda s: s.get("bbox", [0])[0])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                sizes = [span.get("size", 0.0) for span in spans] or [0.0]
                bold = any(span.get("flags", 0) & _BOLD_FLAG for span in spans)
                lines.append(_Line(text=text, size=max(sizes), bold=bold))
        return lines

    def _median_body_size(self, pages: list[list[_Line]]) -> float:
        sizes = [line.size for page in pages for line in page if line.text]
        return statistics.median(sizes) if sizes else 12.0

    @staticmethod
    def _is_heading(line: _Line, body_size: float) -> bool:
        if body_size <= 0:
            return False
        if line.size >= body_size * _HEADING_SIZE_RATIO:
            return True
        return line.bold and line.size >= body_size and len(line.text) <= 100