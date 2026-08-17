"""Reranking benchmark harness: complete, latency-accounted, deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.evaluation.reranking_benchmark import RerankingBenchmarkRunner
from app.embeddings.hash_model import HashEmbeddingModel
from app.ingestion.chunking_registry import load_chunking_registry
from app.reranking.hash_reranker import HashReranker

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runner() -> RerankingBenchmarkRunner:
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    return RerankingBenchmarkRunner(
        corpus=build_chunking_corpus(),
        samples=build_chunking_samples(),
        chunking_registry=registry,
        embedding_model=HashEmbeddingModel(dimension=64),
        reranker=HashReranker(),
    )


@pytest.mark.asyncio
async def test_both_configurations_produce_complete_results() -> None:
    runner = _runner()
    results = await runner.run_all_async()
    assert [r.name for r in results] == ["hybrid_rrf", "hybrid_rrf_reranked"]

    quality_keys = {"mrr", "recall_at_5", "recall_at_10", "precision_at_5", "hit_rate_at_10"}
    latency_keys = {"mean_total_ms", "p95_total_ms", "mean_rerank_ms", "p95_rerank_ms"}
    for result in results:
        assert quality_keys <= set(result.metrics)
        assert latency_keys <= set(result.latency_stats)
        assert all(0.0 <= result.metrics[k] <= 1.0 for k in quality_keys)

    reranked = results[1]
    assert reranked.pairs_scored > 0                    # rerank stage actually ran
    assert reranked.latency_stats["mean_rerank_ms"] >= 0.0
    assert results[0].pairs_scored == 0                 # baseline never scores pairs


@pytest.mark.asyncio
async def test_reranking_benchmark_is_deterministic() -> None:
    first = {r.name: r.metrics for r in await _runner().run_all_async()}
    second = {r.name: r.metrics for r in await _runner().run_all_async()}
    assert first == second