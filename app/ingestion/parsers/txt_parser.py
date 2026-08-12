"""Plain-text parser with a conservative heading heuristic.

TXT has no structure, so we infer headings from typography patterns:
short lines that are ALL CAPS, numbered ("3.1 Title"), or end with ':'
and are followed by a blank line. False positives only cost a section
boundary; missed headings only cost citation granularity.
"""

from __future__ import annotations

import re

from app.ingestion.domain import (
    BlockType,
    ExtractedMetadata,
    ParserOutput,
    StructuralBlock,
)
from app.ingestion.parsers.base import Parser
from app.models.enums import DocumentSourceType

_NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S")
_MAX_HEADING_CHARS = 80


class TxtParser(Parser):
    source_type = DocumentSourceType.TXT
    name = "txt"

    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput:
        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        blocks: list[StructuralBlock] = []
        paragraph: list[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(StructuralBlock(text="\n".join(paragraph)))
                paragraph.clear()

        for i, raw in enumerate(lines):
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue
            followed_by_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
            if followed_by_blank and self._looks_like_heading(stripped):
                flush_paragraph()
                blocks.append(
                    StructuralBlock(
                        text=stripped, block_type=BlockType.HEADING, heading_level=1, page=1
                    )
                )
            else:
                paragraph.append(stripped)
        flush_paragraph()

        title = next((b.text for b in blocks if b.is_heading), None)
        return ParserOutput(
            blocks=blocks,
            metadata=ExtractedMetadata(title=title, page_count=1),
        )

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        if len(line) > _MAX_HEADING_CHARS:
            return False
        if line.isupper() and any(ch.isalpha() for ch in line):
            return True
        if _NUMBERED_RE.match(line):
            return True
        return line.endswith(":")