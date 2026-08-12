"""Markdown parser (line-based, dependency-free).

A full CommonMark engine is unnecessary for ingestion: headings, code
fences, lists, and paragraphs cover what matters for retrieval and
citation. Trade-off documented per Rule 8.
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

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_FENCE_RE = re.compile(r"^```")


class MarkdownParser(Parser):
    source_type = DocumentSourceType.MARKDOWN
    name = "markdown"

    def parse(self, content: bytes, *, source_uri: str) -> ParserOutput:
        text = content.decode("utf-8", errors="replace")
        blocks: list[StructuralBlock] = []
        paragraph: list[str] = []
        code: list[str] = []
        in_code = False
        title: str | None = None

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(StructuralBlock(text="\n".join(paragraph), page=1))
                paragraph.clear()

        for raw in text.splitlines():
            if _FENCE_RE.match(raw.strip()):
                if in_code:
                    blocks.append(StructuralBlock(text="\n".join(code), block_type=BlockType.CODE, page=1))
                    code.clear()
                else:
                    flush_paragraph()
                in_code = not in_code
                continue
            if in_code:
                code.append(raw)
                continue

            heading = _HEADING_RE.match(raw)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                heading_text = heading.group(2).strip()
                if title is None and level == 1:
                    title = heading_text
                blocks.append(
                    StructuralBlock(
                        text=heading_text, block_type=BlockType.HEADING, heading_level=level, page=1
                    )
                )
                continue

            list_item = _LIST_RE.match(raw)
            if list_item:
                flush_paragraph()
                blocks.append(StructuralBlock(text=list_item.group(1).strip(), block_type=BlockType.LIST_ITEM, page=1))
                continue

            if not raw.strip():
                flush_paragraph()
                continue
            paragraph.append(raw.strip())

        flush_paragraph()
        if in_code and code:  # unterminated fence — keep content rather than drop it
            blocks.append(StructuralBlock(text="\n".join(code), block_type=BlockType.CODE, page=1))

        return ParserOutput(
            blocks=blocks, metadata=ExtractedMetadata(title=title, page_count=1)
        )