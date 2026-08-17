"""Fusion engines for hybrid retrieval (PROJECT_SPEC Phase 8).

Two strategies, both pure functions (unit-testable, deterministic):

- Reciprocal Rank Fusion (default, decision D-045):
    score(d) = Σ_list 1 / (k + rank_list(d))
  Rank-based → immune to the scale mismatch between cosine similarity and
  ts_rank_cd/BM25 scores. No calibration needed.

- Weighted fusion:
    score(d) = Σ_list w_list · minmax(score_list(d))
  Min-max normalization per list makes scores comparable; weights express
  trust in each branch. Kept as an experiment arm — it IS sensitive to
  score distributions, which is exactly what benchmarks must reveal.

Chunks appearing in multiple branches are merged once, with the union of
sources and per-branch raw scores preserved for analysis.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Literal

from app.retrieval.base import RetrievedChunk

FusionStrategy = Literal["rrf", "weighted"]


def fuse(
    ranked_lists: Mapping[str, Sequence[RetrievedChunk]],
    strategy: FusionStrategy,
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[RetrievedChunk]:
    if strategy == "rrf":
        return _reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    if strategy == "weighted":
        return _weighted_fusion(ranked_lists, weights or {})
    raise ValueError(f"Unknown fusion strategy: {strategy}")


# --------------------------------------------------------------------- internals
class _Accumulator:
    __slots__ = ("score", "sources", "dense_score", "sparse_score", "best_item", "best_contribution")

    def __init__(self) -> None:
        self.score = 0.0
        self.sources: set[str] = set()
        self.dense_score: float | None = None
        self.sparse_score: float | None = None
        self.best_item: RetrievedChunk | None = None
        self.best_contribution = -1.0

    def add(self, item: RetrievedChunk, contribution: float, origin: str) -> None:
        self.score += contribution
        self.sources.update(item.sources)
        self.sources.add(origin)
        if item.dense_score is not None:
            self.dense_score = item.dense_score
        if item.sparse_score is not None:
            self.sparse_score = item.sparse_score
        if contribution > self.best_contribution:
            self.best_contribution = contribution
            self.best_item = item


def _finalize(merged: dict[uuid.UUID, _Accumulator]) -> list[RetrievedChunk]:
    results: list[tuple[float, RetrievedChunk]] = []
    for acc in merged.values():
        base = acc.best_item
        if base is None:
            continue
        results.append(
            (
                acc.score,
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    version_id=base.version_id,
                    score=acc.score,
                    content=base.content,
                    sources=tuple(sorted(acc.sources)),
                    dense_score=acc.dense_score,
                    sparse_score=acc.sparse_score,
                    section=base.section,
                    source_uri=base.source_uri,
                    title=base.title,
                    page_start=base.page_start,
                    page_end=base.page_end,
                ),
            )
        )
    results.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in results]


def _reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RetrievedChunk]], *, k: int
) -> list[RetrievedChunk]:
    merged: dict[uuid.UUID, _Accumulator] = {}
    for origin, items in ranked_lists.items():
        for rank, item in enumerate(items, start=1):
            contribution = 1.0 / (k + rank)
            merged.setdefault(item.chunk_id, _Accumulator()).add(item, contribution, origin)
    return _finalize(merged)


def _weighted_fusion(
    ranked_lists: Mapping[str, Sequence[RetrievedChunk]], weights: Mapping[str, float]
) -> list[RetrievedChunk]:
    merged: dict[uuid.UUID, _Accumulator] = {}
    for origin, items in ranked_lists.items():
        weight = weights.get(origin, 1.0)
        for item, normalized in _minmax_normalize(items):
            contribution = weight * normalized
            merged.setdefault(item.chunk_id, _Accumulator()).add(item, contribution, origin)
    return _finalize(merged)


def _minmax_normalize(
    items: Sequence[RetrievedChunk],
) -> list[tuple[RetrievedChunk, float]]:
    if not items:
        return []
    scores = [item.score for item in items]
    low, high = min(scores), max(scores)
    span = high - low
    if span == 0:
        return [(item, 1.0) for item in items]
    return [(item, (item.score - low) / span) for item in items]