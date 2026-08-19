#!/usr/bin/env python3
"""Run the Phase 21 retrieval evaluation end-to-end.

This script:
1. Loads the evaluation dataset (datasets/evaluation_dataset.yaml)
2. Ingests the evaluation corpus (datasets/evaluation_corpus/*.md)
3. Builds four retriever configurations (dense, sparse, hybrid, hybrid+reranker)
4. Runs retrieval on all questions and computes metrics
5. Tracks experiments to MLflow (if enabled)
6. Writes docs/retrieval-evaluation.md with the comparison table

Usage:
    python scripts/run_retrieval_evaluation.py

Requirements:
- PostgreSQL running (docker compose up postgres)
- Qdrant running (docker compose up qdrant)
- Evaluation dataset + corpus in datasets/

This is EXPENSIVE (real embeddings, real reranker) — run manually, not in CI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.retrieval_report import render_retrieval_evaluation_report
from app.evaluation.retrieval_runner import RetrievalEvaluationRunner
from app.evaluation.trackers import build_tracker
from app.embeddings.factory import build_embedding_model
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.storage import LocalFileStore
from app.repositories.database import Database
from app.repositories.vector.qdrant_client import build_qdrant_client
from app.repositories.vector.qdrant_repository import QdrantVectorRepository

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "evaluation_dataset.yaml"
CORPUS_DIR = REPO_ROOT / "datasets" / "evaluation_corpus"
REPORT_PATH = REPO_ROOT / "docs" / "retrieval-evaluation.md"

logger = get_logger("scripts.retrieval_evaluation")


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

    qdrant_client = build_qdrant_client(settings.qdrant)
    from app.embeddings.naming import collection_name

    collection = collection_name(settings.qdrant.collection_prefix, settings.embedding.default_model)
    vector_repo = QdrantVectorRepository(qdrant_client, collection)

    # ---- 3. Build embedding service ------------------------------------------
    embedding_registry = load_embedding_registry(settings.embedding.registry_path)
    embedding_model = build_embedding_model(embedding_registry, settings.embedding)
    from app.embeddings.caching import InMemoryCacheStore

    embedding_service = EmbeddingService(
        embedding_model,
        cache=InMemoryCacheStore(),
        max_retries=settings.embedding.max_retries,
    )

    # ---- 4. Build runner -----------------------------------------------------
    chunking_registry = load_chunking_registry()
    file_store = LocalFileStore(REPO_ROOT / "data" / "uploads")

    runner = RetrievalEvaluationRunner(
        database=database,
        vector_repository=vector_repo,
        embedding_service=embedding_service,
        settings=settings,
        chunking_registry=chunking_registry,
        file_store=file_store,
        tracker=build_tracker(settings),
    )

    # ---- 5. Run evaluation ---------------------------------------------------
    logger.info("starting_retrieval_evaluation")
    summaries = await runner.run(dataset, CORPUS_DIR)

    # ---- 6. Print results ----------------------------------------------------
    print("\n" + "=" * 80)
    print("Retrieval Evaluation Results")
    print("=" * 80)
    print(f"\n{'Retriever':<20}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'R@20':>8}{'P@5':>8}{'Hit@10':>8}")
    print("-" * 80)
    for summary in summaries:
        print(
            f"{summary.retriever_name:<20}"
            f"{summary.metrics.get('mrr', 0):>8.4f}"
            f"{summary.metrics.get('recall_at_5', 0):>8.4f}"
            f"{summary.metrics.get('recall_at_10', 0):>8.4f}"
            f"{summary.metrics.get('recall_at_20', 0):>8.4f}"
            f"{summary.metrics.get('precision_at_5', 0):>8.4f}"
            f"{summary.metrics.get('hit_rate_at_10', 0):>8.4f}"
        )
    print("=" * 80)

    # ---- 7. Generate report --------------------------------------------------
    report = render_retrieval_evaluation_report(
        summaries,
        dataset_name=dataset.schema_version,
        num_questions=len(dataset.questions),
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("report_written", path=str(REPORT_PATH))
    print(f"\n✓ Report written to {REPORT_PATH}")

    # ---- 8. Cleanup ----------------------------------------------------------
    await qdrant_client.close()
    await database.dispose()


if __name__ == "__main__":
    asyncio.run(main())