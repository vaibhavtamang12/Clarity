"""Benchmark harness tests: all four configurations, complete + deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.evaluation.retrieval_benchmark import RetrievalBenchmarkRunner
from app.embeddings.hash_model import HashEmbeddingModel
from app.ingestion.chunking_registry import load_chunking_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runner() -> RetrievalBenchmarkRunner:
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    return RetrievalBenchmarkRunner(
        corpus=build_chunking_corpus(),
        samples=build_chunking_samples(),
        chunking_registry=registry,
        embedding_model=HashEmbeddingModel(dimension=64),
    )


@pytest.mark.asyncio
async def test_all_configurations_produce_complete_metrics() -> None:
    runner = _runner()
    results = await runner.run_all_async()
    assert [r.name for r in results] == ["dense", "sparse_bm25", "hybrid_weighted", "hybrid_rrf"]
    for result in results:
        for key in ("mrr", "recall_at_5", "recall_at_10", "precision_at_5", "hit_rate_at_10"):
            assert key in result.metrics
            assert 0.0 <= result.metrics[key] <= 1.0
        assert len(result.per_sample_gold_sizes) == len(runner.samples)


@pytest.mark.asyncio
async def test_hybrid_never_worse_than_worst_single_branch_on_hit_rate() -> None:
    """Sanity invariant: fusing two branches cannot produce ZERO hits where both
    branches individually found the evidence. (Recall can vary; total failure
    cannot be introduced by fusion.)"""
    runner = _runner()
    results = {r.name: r for r in await runner.run_all_async()}
    assert results["hybrid_rrf"].metrics["hit_rate_at_10"] >= min(
        results["dense"].metrics["hit_rate_at_10"],
        results["sparse_bm25"].metrics["hit_rate_at_10"],
    ) - 1e-9


@pytest.mark.asyncio
async def test_benchmark_is_deterministic() -> None:
    first = {r.name: r.metrics for r in await _runner().run_all_async()}
    second = {r.name: r.metrics for r in await _runner().run_all_async()}
    assert first == second