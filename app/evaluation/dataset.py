"""Evaluation dataset loader + validator (Phase 20).

Loads YAML datasets, validates against the schema, and provides utilities
for running evaluations. The dataset is separate from the harness code
(D-110): data files live in datasets/, not hardcoded in Python.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.evaluation.dataset_schema import EvaluationDataset, EvaluationQuestion

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path("datasets/evaluation_dataset.yaml")


def load_evaluation_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> EvaluationDataset:
    """Load and validate an evaluation dataset from YAML."""
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Evaluation dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    try:
        dataset = EvaluationDataset.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid evaluation dataset at {path}: {exc}") from exc

    logger.info(
        "evaluation_dataset_loaded",
        path=str(path),
        schema_version=dataset.schema_version,
        corpus_version=dataset.corpus_version,
        num_questions=len(dataset.questions),
    )
    return dataset


def get_questions_by_category(
    dataset: EvaluationDataset, category: str
) -> list[EvaluationQuestion]:
    """Filter questions by taxonomy category."""
    return [q for q in dataset.questions if q.category == category]


def validate_corpus_coverage(dataset: EvaluationDataset, corpus_files: set[str]) -> list[str]:
    """Check that all expected_sources reference documents in the corpus.

    Returns a list of warning messages (empty if all sources exist).
    """
    warnings: list[str] = []
    for question in dataset.questions:
        for source in question.expected_sources:
            if source not in corpus_files:
                warnings.append(
                    f"Question {question.id}: expected source '{source}' not in corpus"
                )
    return warnings


def compute_dataset_stats(dataset: EvaluationDataset) -> dict[str, int | float]:
    """Compute dataset statistics (category distribution, difficulty, etc.)."""
    stats: dict[str, int | float] = {
        "total_questions": len(dataset.questions),
        "unique_documents": len(
            {source for q in dataset.questions for source in q.expected_sources}
        ),
    }

    # Category distribution
    for category in [
        "simple_factual",
        "multi_hop",
        "ambiguous",
        "unanswerable",
        "temporal_version",
        "metadata_filter",
    ]:
        count = len(get_questions_by_category(dataset, category))
        stats[f"category_{category}"] = count
        stats[f"pct_{category}"] = round(100 * count / len(dataset.questions), 1)

    # Difficulty distribution
    for difficulty in ["easy", "medium", "hard"]:
        count = len([q for q in dataset.questions if q.difficulty == difficulty])
        stats[f"difficulty_{difficulty}"] = count

    return stats