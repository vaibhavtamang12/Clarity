#!/usr/bin/env python3
"""Run the Phase 9 reranking benchmark (without vs with reranker).

Writes docs/reranking-experiments.md with measured numbers and logs each
configuration to MLflow when enabled.

Usage:
    python scripts/run_reranking_benchmark.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.evaluation.reranking_benchmark import RerankingBenchmarkRunner
from app.evaluation.reranking_report import render_reranking_report
from app.evaluation.trackers import build_tracker
from app.embeddings.factory import build_embedding_model
from app.embeddings.registry import load_embedding_registry
from app.ingestion.chunking_registry import load_chunking_registry
from app.reranking.factory import build_reranker
from app.reranking.registry import load_reranker_registry

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = get_logger("scripts.reranking_benchmark")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    chunking_registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    embedding_registry = load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")
    reranker_registry = load_reranker_registry(REPO_ROOT / "configs" / "rerankers.yaml")

    embedding_model = build_embedding_model(embedding_registry, settings.embedding, model_key="hash_64")
    reranker = build_reranker(reranker_registry, settings.reranker, model_key="hash")

    runner = RerankingBenchmarkRunner(
        corpus=build_chunking_corpus(),
        samples=build_chunking_samples(),
        chunking_registry=chunking_registry,
        embedding_model=embedding_model,
        reranker=reranker,
        tracker=build_tracker(settings),
    )
    results = await runner.run_all_async()

    print()
    print(f"{'configuration':<22}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'mean ms':>10}{'pairs':>8}")
    for r in results:
        print(
            f"{r.name:<22}"
            f"{r.metrics['mrr']:>8.4f}"
            f"{r.metrics['recall_at_5']:>8.4f}"
            f"{r.metrics['recall_at_10']:>8.4f}"
            f"{r.latency_stats['mean_total_ms']:>10.1f}"
            f"{r.pairs_scored:>8}"
        )

    output = REPO_ROOT / "docs" / "reranking-experiments.md"
    output.write_text(
        render_reranking_report(results, n_samples=len(runner.samples), reranker_key="hash"),
        encoding="utf-8",
    )
    logger.info("report_written", path=str(output))


if __name__ == "__main__":
    asyncio.run(main())