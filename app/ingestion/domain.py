"""Intermediate representation shared by all parsers and chunkers.

Every parser — regardless of source format — emits a flat list of
StructuralBlocks. Everything downstream (cleaning, structure annotation,
chunking) works only on this IR. Adding a new file format means
implementing one interface; nothing else changes (decision D-023).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class BlockType(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE = "code"
    LIST_ITEM = "list_item"


@dataclass
class StructuralBlock:
    text: str
    block_type: BlockType = BlockType.PARAGRAPH
    page: int | None = None
    heading_level: int = 0  # 1..6 for headings, 0 otherwise
    section_path: tuple[str, ...] = ()  # filled by the structure annotator

    @property
    def is_heading(self) -> bool:
        return self.block_type == BlockType.HEADING


@dataclass
class ExtractedMetadata:
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ParserOutput:
    blocks: list[StructuralBlock]
    metadata: ExtractedMetadata


@dataclass(frozen=True)
class Chunk:
    """A chunk with everything precise citation requires (PROJECT_SPEC FR-2.3)."""

    text: str
    chunk_index: int
    token_count: int
    char_count: int
    content_hash: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()