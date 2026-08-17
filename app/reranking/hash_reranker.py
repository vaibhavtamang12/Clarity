"""Deterministic token-overlap reranker — the reranking test double (D-051).

Scores the normalized token overlap between query and chunk. Good enough to
exercise ordering changes, candidate cuts, and latency accounting end-to-end
in CI with zero model downloads. Quality deltas of the real cross-encoder
are measured in Phase 21 (D-052).
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.base import RetrievedChunk
from app.retrieval.bm25 import tokenize


class HashReranker:
    model_id = "hash"

    def __init__(self, model_key: str = "hash") -> None:
        self.model_key = model_key

    def rerank(self, query: str, items: Sequence[RetrievedChunk]) -> list[float]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [0.0 for _ in items]
        scores: list[float] = []
        for item in items:
            chunk_tokens = set(tokenize(item.content))
            scores.append(len(query_tokens & chunk_tokens) / len(query_tokens))
        return scores