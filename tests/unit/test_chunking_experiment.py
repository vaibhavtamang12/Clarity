"""Harness-level tests: the runner executes the real pipeline end-to-end
(in-memory, no DB, no MLflow) and produces complete, deterministic results."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.runner import ChunkingExperimentRunner
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.ingestion.chunking_registry import load_chunking_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runner():
    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    return ChunkingExperimentRunner(
        corpus=build_chunking_corpus(),
        samples=build_chunking_samples(),
        chunking_registry=registry,
    )


def test_run_all_strategies_produces_complete_results() -> None:
    runner = _runner()
    results = runner.run_all()
    assert len(results) == len(runner.registry.strategies)
    for result in results:
        for key in ("mrr", "recall_at_5", "recall_at_10", "precision_at_5", "precision_at_10"):
            assert key in result.metrics
            assert 0.0 <= result.metrics[key] <= 1.0
        assert result.stats.n_chunks > 0
        assert len(result.samples) == len(runner.samples)


def test_every_question_has_gold_evidence_in_the_corpus() -> None:
    """Sanity: no sample should have zero gold chunks under a generous strategy —
    that would mean the evidence span is missing from the corpus itself."""
    runner = _runner()
    result = runner.run_strategy("structure_aware_512")
    orphans = [s.sample_id for s in result.samples if s.n_relevant_chunks == 0]
    assert orphans == [], f"samples with no gold chunks: {orphans}"


def test_experiment_is_deterministic() -> None:
    runner = _runner()
    first = runner.run_strategy("recursive_512").metrics
    second = runner.run_strategy("recursive_512").metrics
    assert first == second