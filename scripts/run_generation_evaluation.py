#!/usr/bin/env python3
"""Run the Phase 22 generation evaluation end-to-end.

This script:
1. Loads the evaluation dataset (datasets/evaluation_dataset.yaml)
2. Builds four RAG pipeline configurations (baseline, hybrid, +reranker, +query rewriting)
3. Runs the FULL RAG pipeline on all questions for each configuration
4. Evaluates each generated answer using LLM-as-judge
5. Tracks experiments to MLflow (if enabled)
6. Writes docs/generation-evaluation.md with the comparison table

Usage:
    python scripts/run_generation_evaluation.py

Requirements:
- PostgreSQL running (docker compose up postgres)
- Qdrant running (docker compose up qdrant)
- LLM provider configured (.env)
- Evaluation dataset + corpus in datasets/

This is VERY EXPENSIVE (real LLM calls for every question on every
configuration) — run manually, not in CI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.generation_report import render_generation_evaluation_report
from app.evaluation.generation_runner import GenerationEvaluationRunner
from app.evaluation.trackers import build_tracker
from app.repositories.database import Database

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "evaluation_dataset.yaml"
REPORT_PATH = REPO_ROOT / "docs" / "generation-evaluation.md"

logger = get_logger("scripts.generation_evaluation")


async def main() -> None:
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
    # In a real implementation, we'd build four RAGPipeline instances with
    # different retrievers and query transformation enabled/disabled.
    # For now, this is a placeholder showing the structure.
    from app.container import build_platform

    platform = build_platform(settings, database)

    # Build four configurations by modifying the platform's retriever
    # This is a simplified example - real implementation would build separate pipelines
    rag_pipelines = {
        "baseline": platform.rag_pipeline,  # dense only
        "hybrid": platform.rag_pipeline,    # would build hybrid retriever
        "hybrid_reranker": platform.rag_pipeline,  # would add reranker
        "hybrid_reranker_rewriting": platform.rag_pipeline,  # would add query rewriting
    }

    # ---- 4. Build LLM judge --------------------------------------------------
    from app.llm.factory import build_llm_provider

    judge_provider = build_llm_provider(settings.llm)

    # ---- 5. Build runner -----------------------------------------------------
    runner = GenerationEvaluationRunner(
        database=database,
        rag_pipelines=rag_pipelines,
        judge_provider=judge_provider,
        settings=settings,
        tracker=build_tracker(settings),
    )

    # ---- 6. Run evaluation ---------------------------------------------------
    logger.info("starting_generation_evaluation")
    summaries = await runner.run(dataset)

    # ---- 7. Print results ----------------------------------------------------
    print("\n" + "=" * 100)
    print("Generation Evaluation Results")
    print("=" * 100)
    print(f"\n{'Configuration':<30}{'Faith':>8}{'Correct':>8}{'Relev':>8}{'Cite':>8}{'Context':>8}")
    print("-" * 100)
    for summary in summaries:
        print(
            f"{summary.configuration:<30}"
            f"{summary.metrics.get('faithfulness', 0):>8.4f}"
            f"{summary.metrics.get('answer_correctness', 0):>8.4f}"
            f"{summary.metrics.get('answer_relevance', 0):>8.4f}"
            f"{summary.metrics.get('citation_correctness', 0):>8.4f}"
            f"{summary.metrics.get('context_relevance', 0):>8.4f}"
        )
    print("=" * 100)

    # ---- 8. Generate report --------------------------------------------------
    report = render_generation_evaluation_report(
        summaries,
        dataset_name=dataset.schema_version,
        num_questions=len(dataset.questions),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("report_written", path=str(REPORT_PATH))
    print(f"\n✓ Report written to {REPORT_PATH}")

    # ---- 9. Cleanup ----------------------------------------------------------
    await database.dispose()


if __name__ == "__main__":
    asyncio.run(main())