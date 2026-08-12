#!/usr/bin/env python3
"""Run the Phase 5 chunking experiments end-to-end.

- Chunks the corpus with every registered strategy (real ingestion path)
- Scores each strategy with BM25 retrieval over gold evidence spans
- Logs each strategy run to MLflow (when enabled)
- Writes docs/chunking-experiments.md with the measured numbers

Usage:
    python scripts/run_chunking_experiments.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.chunking_eval.corpus import build_chunking_corpus
from app.evaluation.chunking_eval.report import write_report
from app.evaluation.chunking_eval.runner import (
    DEFAULT_K_VALUES,
    DEFAULT_RETRIEVAL_DEPTH,
    ChunkingExperimentRunner,
)
from app.evaluation.chunking_eval.samples import build_chunking_samples
from app.evaluation.trackers import build_tracker
from app.ingestion.chunking_registry import load_chunking_registry

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = get_logger("scripts.chunking_experiments")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    registry = load_chunking_registry(REPO_ROOT / "configs" / "chunking.yaml")
    corpus = build_chunking_corpus()
    samples = build_chunking_samples()
    tracker = build_tracker(settings)

    runner = ChunkingExperimentRunner(
        corpus=corpus,
        samples=samples,
        chunking_registry=registry,
        tracker=tracker,
    )
    results = runner.run_all()

    print()
    print(f"{'strategy':<18}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'P@5':>8}{'chunks':>8}")
    for r in results:
        print(
            f"{r.strategy:<18}"
            f"{r.metrics['mrr']:>8.4f}"
            f"{r.metrics['recall_at_5']:>8.4f}"
            f"{r.metrics['recall_at_10']:>8.4f}"
            f"{r.metrics['precision_at_5']:>8.4f}"
            f"{r.stats.n_chunks:>8}"
        )

    output = write_report(
        results,
        output_path=REPO_ROOT / "docs" / "chunking-experiments.md",
        corpus_size=len(corpus),
        n_samples=len(samples),
        retrieval_depth=DEFAULT_RETRIEVAL_DEPTH,
        k_values=DEFAULT_K_VALUES,
    )
    logger.info("report_written", path=str(output))


if __name__ == "__main__":
    asyncio.run(main())