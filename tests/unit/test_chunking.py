"""Unit tests for all four chunking strategies + the factory."""

from __future__ import annotations

import pytest

from app.core.exceptions import ConfigurationError
from app.ingestion.chunking.factory import build_chunker
from app.ingestion.chunking_registry import ChunkingStrategyConfig
from app.ingestion.domain import BlockType, StructuralBlock
from app.ingestion.structure import annotate_sections
from app.ingestion.domain import ParserOutput, ExtractedMetadata

LONG_PARAGRAPH = "The refund policy applies to all enterprise customers. " * 40


def _config(strategy: str, **overrides: int) -> ChunkingStrategyConfig:
    base = {"type": strategy, "target_tokens": 50, "overlap_tokens": 10, "min_tokens": 10, "max_tokens": 100}
    base.update(overrides)
    return ChunkingStrategyConfig.model_validate(base)


def _blocks() -> list[StructuralBlock]:
    return [
        StructuralBlock(text="Section A heading", block_type=BlockType.HEADING, heading_level=1, page=1),
        StructuralBlock(text="First paragraph under A.", page=1),
        StructuralBlock(text="Section B heading", block_type=BlockType.HEADING, heading_level=1, page=2),
        StructuralBlock(text="First paragraph under B.", page=2),
    ]


def _annotated() -> list[StructuralBlock]:
    output = annotate_sections(
        ParserOutput(blocks=_blocks(), metadata=ExtractedMetadata(title="T"))
    )
    return output.blocks


def test_fixed_chunker_respects_budget_and_overlap() -> None:
    chunker = build_chunker(_config("fixed"))
    chunks = chunker.chunk([StructuralBlock(text=LONG_PARAGRAPH, page=1)])
    assert len(chunks) >= 2
    assert all(c.token_count <= 100 for c in chunks)
    # overlap: the tail of chunk N reappears at the head of chunk N+1
    assert chunks[0].text[-20:] in chunks[1].text
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_recursive_chunker_splits_long_text() -> None:
    chunker = build_chunker(_config("recursive"))
    chunks = chunker.chunk([StructuralBlock(text=LONG_PARAGRAPH, page=1)])
    assert len(chunks) >= 2
    assert all(c.token_count <= 100 for c in chunks)


def test_sentence_chunker_keeps_sentence_boundaries() -> None:
    text = "One sentence here. Another sentence there. A third one follows."
    chunker = build_chunker(_config("sentence", target_tokens=8, max_tokens=12, overlap_tokens=0))
    chunks = chunker.chunk([StructuralBlock(text=text, page=1)])
    for chunk in chunks:
        # every chunk is a union of full sentences
        assert not chunk.text.strip().rstrip(".!?").split()[-1].isalpha() or chunk.text.strip().endswith((".", "!", "?"))


def test_structure_aware_never_merges_across_headings() -> None:
    chunker = build_chunker(_config("structure_aware", target_tokens=500))
    chunks = chunker.chunk(_annotated())
    sections = {c.section for c in chunks}
    assert sections == {"Section A heading", "Section B heading"}
    for chunk in chunks:
        # no chunk contains text from both sections
        assert not ("under A" in chunk.text and "under B" in chunk.text)


def test_structure_aware_carries_citation_metadata() -> None:
    chunker = build_chunker(_config("structure_aware", target_tokens=500))
    chunks = chunker.chunk(_annotated())
    section_b = next(c for c in chunks if c.section == "Section B heading")
    assert section_b.page_start == 2
    assert section_b.heading_path == ["Section B heading"]
    assert section_b.content_hash and section_b.char_count > 0


def test_chunk_hashes_are_deterministic() -> None:
    chunker = build_chunker(_config("structure_aware", target_tokens=500))
    first = [c.content_hash for c in chunker.chunk(_annotated())]
    second = [c.content_hash for c in chunker.chunk(_annotated())]
    assert first == second


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ConfigurationError):
        build_chunker(_config("quantum"))