"""Deterministic text cleaning applied to every parsed block.

Order matters and is fixed: normalize → strip control/zero-width chars →
repair line-break hyphenation → collapse whitespace. Determinism is a
requirement: identical input bytes must always produce identical chunks
(idempotent re-ingestion depends on it).
"""

from __future__ import annotations

import re
import unicodedata

from app.ingestion.domain import ParserOutput, StructuralBlock

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_RUNS_OF_SPACES_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    # "exam-\nple" → "example" (keep genuine hyphens like "state-of-the-art")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [_RUNS_OF_SPACES_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_parsed_document(output: ParserOutput) -> ParserOutput:
    cleaned: list[StructuralBlock] = []
    for block in output.blocks:
        text = clean_text(block.text)
        if not text:
            continue  # empty blocks carry no information and would pollute chunks
        cleaned.append(
            StructuralBlock(
                text=text,
                block_type=block.block_type,
                page=block.page,
                heading_level=block.heading_level,
                section_path=block.section_path,
            )
        )
    return ParserOutput(blocks=cleaned, metadata=output.metadata)