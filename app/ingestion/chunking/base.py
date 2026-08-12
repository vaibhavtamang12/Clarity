"""Chunker interface, Chunk unit, and the shared packing engine.

All strategies share one engine (accumulate units until the token budget is
reached, split oversized units, carry overlap forward). Strategies differ in
two hooks only:
  - respect_sections: never merge across heading boundaries
  - split_oversized:  how a too-large unit is broken down
This keeps the four strategies comparable in Phase 5 benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.ingestion.chunking.common import hard_split, recursive_split
from app.ingestion.domain import Chunk, StructuralBlock, sha256_hex
from app.ingestion.chunking_registry import ChunkingStrategyConfig
from app.ingestion.tokens import TokenCounter

_JOIN = "\n\n"


class _Unit:
    __slots__ = ("text", "page", "section_path")

    def __init__(self, text: str, page: int | None, section_path: tuple[str, ...]) -> None:
        self.text = text
        self.page = page
        self.section_path = section_path


class Chunker(ABC):
    def __init__(self, config: ChunkingStrategyConfig, token_counter: TokenCounter) -> None:
        self.config = config
        self.counter = token_counter

    @abstractmethod
    def chunk(self, blocks: Sequence[StructuralBlock]) -> list[Chunk]: ...


class PackingChunker(Chunker):
    respect_sections: bool = False

    def split_oversized(self, text: str) -> list[str]:
        return recursive_split(text, self.config.max_tokens, self.counter)

    # ------------------------------------------------------------------ engine
    def chunk(self, blocks: Sequence[StructuralBlock]) -> list[Chunk]:
        units = [_Unit(b.text, b.page, b.section_path) for b in blocks if b.text.strip()]
        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_units: list[_Unit] = []
        carry = ""

        def joined() -> str:
            return _JOIN.join(current_texts)

        def close() -> None:
            nonlocal carry
            if not current_texts:
                return
            text = joined()
            chunks.append(self._make_chunk(text, current_units, len(chunks)))
            overlap_chars = self.config.overlap_tokens * 4  # heuristic chars/token
            carry = text[-overlap_chars:] if self.config.overlap_tokens > 0 else ""
            current_texts.clear()
            current_units.clear()

        for unit in units:
            if (
                self.respect_sections
                and current_units
                and unit.section_path != current_units[-1].section_path
            ):
                close()

            if carry and not current_texts:
                current_texts.append(carry)
                carry = ""

            candidate = _JOIN.join([*current_texts, unit.text])
            if not current_units or self.counter.count(candidate) <= self.config.target_tokens:
                current_texts.append(unit.text)
                current_units.append(unit)
                if self.counter.count(joined()) >= self.config.target_tokens:
                    close()
                continue

            # Unit overflows the budget: close current chunk, split the unit.
            close()
            if carry:
                current_texts.append(carry)
                carry = ""
            for piece in self.split_oversized(unit.text):
                for sub in hard_split(piece, self.config.max_tokens, self.counter):
                    current_texts.append(sub)
                    current_units.append(unit)
                    if self.counter.count(joined()) >= self.config.target_tokens:
                        close()
                        if carry:
                            current_texts.append(carry)
                            carry = ""
        close()
        return self._apply_minimum(chunks)

    # ------------------------------------------------------------------ helpers
    def _make_chunk(self, text: str, units: Sequence[_Unit], index: int) -> Chunk:
        pages = [u.page for u in units if u.page is not None]
        first_path = units[0].section_path if units else ()
        return Chunk(
            text=text,
            chunk_index=index,
            token_count=self.counter.count(text),
            char_count=len(text),
            content_hash=sha256_hex(text),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section=first_path[-1] if first_path else None,
            heading_path=list(first_path),
        )

    def _apply_minimum(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge an undersized trailing chunk into its predecessor when safe."""
        if len(chunks) < 2:
            return chunks
        last = chunks[-1]
        if last.token_count >= self.config.min_tokens:
            return chunks
        previous = chunks[-2]
        if self.respect_sections and last.section != previous.section:
            return chunks  # never merge across sections — citation integrity first
        merged_text = f"{previous.text}\n\n{last.text}"
        chunks[-2] = Chunk(
            text=merged_text,
            chunk_index=previous.chunk_index,
            token_count=self.counter.count(merged_text),
            char_count=len(merged_text),
            content_hash=sha256_hex(merged_text),
            page_start=min(p for p in (previous.page_start, last.page_start) if p is not None)
            if any(p is not None for p in (previous.page_start, last.page_start))
            else None,
            page_end=max(p for p in (previous.page_end, last.page_end) if p is not None)
            if any(p is not None for p in (previous.page_end, last.page_end))
            else None,
            section=previous.section,
            heading_path=previous.heading_path,
        )
        return chunks[:-1]