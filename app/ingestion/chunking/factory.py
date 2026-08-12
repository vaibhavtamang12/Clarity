# app/ingestion/chunking/factory.py
"""Chunker factory — config in, chunker out. Pipelines never construct
strategy classes directly, which keeps Phase 5 experiments config-driven."""

from __future__ import annotations

from app.core.exceptions import ConfigurationError
from app.ingestion.chunking.base import Chunker
from app.ingestion.chunking.strategies import (
    FixedChunker,
    RecursiveChunker,
    SentenceChunker,
    StructureAwareChunker,
)
from app.ingestion.chunking_registry import ChunkingStrategyConfig
from app.ingestion.tokens import HeuristicTokenCounter, TokenCounter

_STRATEGIES = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "structure_aware": StructureAwareChunker,
}


def build_chunker(
    config: ChunkingStrategyConfig, token_counter: TokenCounter | None = None
) -> Chunker:
    cls = _STRATEGIES.get(config.type)
    if cls is None:
        raise ConfigurationError(f"Unknown chunking strategy: {config.type}")
    return cls(config, token_counter or HeuristicTokenCounter())