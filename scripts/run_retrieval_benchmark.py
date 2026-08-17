#!/usr/bin/env python3
"""Run the Phase 8 retrieval benchmark end-to-end.

Compares dense / sparse / hybrid(weighted) / hybrid(rrf) on the shared
chunking corpus + gold samples, logs each configuration to MLflow when
enabled, and writes docs/retrieval-benchmarks.md with measured numbers.

Usage:
    python scripts/run_retrieval_benchmark.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.evaluation.retrieval_benchmark import (
    DEFAULT_K_VALUES,
    RetrievalBenchmarkRunner,
)
from app.evaluation.retrieval_report import render_retrieval_report
from app.evaluation.trackers import build_tracker
from app.embeddings.factory import build_embedding_model
from app.embeddings.registry import load_embedding_registry
from app.ingestion.chunking_registry import load_chunking_registry

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = get_logger("scripts.retrieval_benchmark")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    chunking_registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    embedding_registry = load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")
    # Hash model keeps the benchmark infrastructure-free and deterministic (D-036).
    embedding_model = build_embedding_model(embedding_registry, settings.embedding, model_key="hash_64")

    corpus = build_chunking_corpus()
    samples = build_chunking_samples()

    runner = RetrievalBenchmarkRunner(
        corpus=corpus,
        samples=samples,
        chunking_registry=chunking_registry,
        embedding_model=embedding_model,
        tracker=build_tracker(settings),
    )
    results = await runner.run_all_async()

    print()
    print(f"{'retriever':<18}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'P@5':>8}")
    for r in results:
        print(
            f"{r.name:<18}"
            f"{r.metrics['mrr']:>8.4f}"
            f"{r.metrics['recall_at_5']:>8.4f}"
            f"{r.metrics['recall_at_10']:>8.4f}"
            f"{r.metrics['precision_at_5']:>8.4f}"
        )

    output = REPO_ROOT / "docs" / "retrieval-benchmarks.md"
    output.write_text(
        render_retrieval_report(
            results,
            n_samples=len(samples),
            n_chunks=len(runner.indexed.texts),
            k_values=DEFAULT_K_VALUES,
        ),
        encoding="utf-8",
    )
    logger.info("report_written", path=str(output))


if __name__ == "__main__":
    asyncio.run(main())