"""Hand-computed correctness tests for retrieval metrics."""

from __future__ import annotations

from app.evaluation.metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = [1, 5, 2, 8]
RELEVANT = {2, 8, 9}


def test_precision_at_k() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.0
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5
    assert precision_at_k([], RELEVANT, 4) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.0
    assert abs(recall_at_k(RANKED, RELEVANT, 4) - 2 / 3) < 1e-9
    # empty gold set → 0.0 by convention (documented)
    assert recall_at_k(RANKED, set(), 4) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == 1 / 3
    assert reciprocal_rank([7, 7], RELEVANT) == 0.0
    assert reciprocal_rank([2], RELEVANT) == 1.0


def test_hit_at_k() -> None:
    assert hit_at_k(RANKED, RELEVANT, 2) == 0.0
    assert hit_at_k(RANKED, RELEVANT, 3) == 1.0


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0