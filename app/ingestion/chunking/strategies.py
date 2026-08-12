# app/ingestion/chunking/strategies.py
"""Concrete chunking strategies (ADR-003, benchmarked in Phase 5)."""

from __future__ import annotations

from app.ingestion.chunking.base import PackingChunker
from app.ingestion.chunking.common import hard_split, recursive_split, split_sentences
from app.ingestion.tokens import TokenCounter


class FixedChunker(PackingChunker):
    """Fixed-size by tokens. Ignores structure — the baseline for experiments."""

    respect_sections = False

    def split_oversized(self, text: str) -> list[str]:
        return hard_split(text, self.config.max_tokens, self.counter)


class RecursiveChunker(PackingChunker):
    """Recursive separator split (\\n\\n → \\n → sentence → word)."""

    respect_sections = False


class SentenceChunker(PackingChunker):
    """Packs whole sentences; sentence boundaries are never broken mid-way
    unless a single sentence exceeds max_tokens."""

    respect_sections = False

    def split_oversized(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        return sentences if sentences else [text]


class StructureAwareChunker(PackingChunker):
    """Default strategy (ADR-003): never merges across headings, so every
    chunk maps cleanly to a citable section."""

    respect_sections = True