"""Retrieval metrics — the Phase 21 metric set, built once and reused.

Conventions (documented to keep every experiment comparable):
- `ranked` is an ordered list of chunk indices, best first.
- `relevant` is the gold set of chunk indices for one sample.
- If the gold set is empty for a sample (evidence shredded beyond recall),
  recall is defined as 0.0 — the strategy is penalized, which is correct.
"""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for idx in top_k if idx in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    return sum(1 for idx in top_k if idx in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for position, idx in enumerate(ranked, start=1):
        if idx in relevant:
            return 1.0 / position
    return 0.0


def hit_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(idx in relevant for idx in ranked[:k]) else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0