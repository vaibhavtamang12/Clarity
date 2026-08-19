#!/usr/bin/env python3
"""Run the Phase 23 latency and cost benchmark end-to-end.

This script:
1. Loads the evaluation dataset (datasets/evaluation_dataset.yaml)
2. Builds RAG pipeline configurations
3. Runs each query through the pipeline multiple times (warmup + measurement)
4. Collects per-stage latency samples and computes p50/p95/p99
5. Tracks token usage and estimates cost per query
6. Writes docs/latency-cost-benchmark.md with the results

Usage:
    python scripts/run_latency_benchmark.py [--iterations 3]

Requirements:
- PostgreSQL running (docker compose up postgres)
- Qdrant running (docker compose up qdrant)
- LLM provider configured (.env)
- Evaluation dataset + corpus in datasets/

This is EXPENSIVE (real LLM calls × iterations) — run manually, not in CI.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.latency_benchmark import LatencyBenchmarkRunner
from app.evaluation.latency_report import render_latency_benchmark_report
from app.evaluation.trackers import build_tracker
from app.repositories.database import Database

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "evaluation_dataset.yaml"
REPORT_PATH = REPO_ROOT / "docs" / "latency-cost-benchmark.md"

logger = get_logger("scripts.latency_benchmark")


async def main(iterations: int = 3, warmup: int = 1) -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    # ---- 1. Load dataset -----------------------------------------------------
    logger.info("loading_evaluation_dataset", path=str(DATASET_PATH))
    dataset = load_evaluation_dataset(DATASET_PATH)
    logger.info(
        "dataset_loaded",
        schema_version=dataset.schema_version,
        num_questions=len(dataset.questions),
    )

    # ---- 2. Initialize infrastructure ----------------------------------------
    logger.info("initializing_infrastructure")
    database = Database(settings.database)
    await database.initialize()

    # ---- 3. Build RAG pipeline configurations --------------------------------
    from app.container import build_platform

    platform = build_platform(settings, database)

    # Build configurations to benchmark
    # In production, you'd build separate pipelines with different retrievers
    # For now, benchmark the default pipeline
    rag_pipelines = {
        "default": platform.rag_pipeline,
    }

    # ---- 4. Build runner -----------------------------------------------------
    runner = LatencyBenchmarkRunner(
        database=database,
        rag_pipelines=rag_pipelines,
        settings=settings,
        tracker=build_tracker(settings),
    )

    # ---- 5. Run benchmark ----------------------------------------------------
    logger.info(
        "starting_latency_benchmark",
        iterations=iterations,
        warmup=warmup,
    )
    results = await runner.run(dataset, iterations=iterations, warmup_iterations=warmup)

    # ---- 6. Print results ----------------------------------------------------
    print("\n" + "=" * 100)
    print("Latency and Cost Benchmark Results")
    print("=" * 100)

    for result in results:
        print(f"\n{result.configuration}")
        print("-" * 100)

        stats = result.total_latency_stats
        print(f"  Total latency: mean={stats.mean_ms:.1f}ms, p50={stats.p50_ms:.1f}ms, p95={stats.p95_ms:.1f}ms, p99={stats.p99_ms:.1f}ms")

        token_stats = result.token_stats
        print(f"  Token usage: avg={token_stats.avg_tokens_per_query:.1f} tokens/query")
        print(f"  Cost: ${token_stats.avg_cost_per_query_usd:.6f}/query, ${token_stats.total_cost_usd:.6f} total")

    print("=" * 100)

    # ---- 7. Generate report --------------------------------------------------
    report = render_latency_benchmark_report(
        results,
        dataset_name=dataset.schema_version,
        iterations=iterations,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("report_written", path=str(REPORT_PATH))
    print(f"\n✓ Report written to {REPORT_PATH}")

    # ---- 8. Cleanup ----------------------------------------------------------
    await database.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 23 latency and cost benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Measurement iterations per query")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per query")
    args = parser.parse_args()
    asyncio.run(main(iterations=args.iterations, warmup=args.warmup))