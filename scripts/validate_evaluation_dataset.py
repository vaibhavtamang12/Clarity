#!/usr/bin/env python3
"""Validate the evaluation dataset for quality and consistency.

Checks:
1. Schema validation (Pydantic)
2. Corpus coverage (all expected_sources exist in corpus)
3. Evidence spans are verbatim in corpus (containment or token overlap)
4. Category distribution is balanced (no category < 5% of questions)
5. Unanswerable questions have empty evidence and reference "I don't know"

Usage:
    python scripts/validate_evaluation_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.logging import setup_logging
from app.evaluation.dataset import (
    compute_dataset_stats,
    load_evaluation_dataset,
    validate_corpus_coverage,
)
from app.evaluation.relevance import evidence_contained, token_overlap_ratio

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "evaluation_dataset.yaml"
CORPUS_DIR = REPO_ROOT / "datasets" / "evaluation_corpus"

MIN_CATEGORY_PCT = 5.0


def load_corpus_text() -> dict[str, str]:
    """Load all corpus documents as text."""
    corpus: dict[str, str] = {}
    if not CORPUS_DIR.exists():
        return corpus
    for path in CORPUS_DIR.glob("*.md"):
        corpus[path.name] = path.read_text(encoding="utf-8")
    return corpus


def check_evidence_in_corpus(dataset, corpus: dict[str, str]) -> list[str]:
    """Verify each evidence span exists in the referenced document."""
    warnings: list[str] = []
    for question in dataset.questions:
        for i, evidence in enumerate(question.evidence):
            doc_text = corpus.get(evidence.document)
            if doc_text is None:
                warnings.append(
                    f"{question.id}: evidence[{i}] references missing document '{evidence.document}'"
                )
                continue
            if evidence_contained(doc_text, evidence.text):
                continue
            overlap = token_overlap_ratio(doc_text, evidence.text)
            if overlap < 0.7:
                warnings.append(
                    f"{question.id}: evidence[{i}] not found in '{evidence.document}' "
                    f"(overlap={overlap:.2f})"
                )
    return warnings


def check_unanswerable_format(dataset) -> list[str]:
    """Unanswerable questions must have empty evidence and say so in the reference."""
    warnings: list[str] = []
    for question in dataset.questions:
        if question.category == "unanswerable":
            if question.evidence:
                warnings.append(f"{question.id}: unanswerable but has evidence")
            answer_lower = question.reference_answer.lower()
            if "not" not in answer_lower and "no evidence" not in answer_lower:
                warnings.append(
                    f"{question.id}: unanswerable reference should state absence of evidence"
                )
    return warnings


def main() -> None:
    setup_logging("INFO", json_output=False)

    print("=" * 80)
    print("Evaluation Dataset Validation")
    print("=" * 80)

    # ---- 1. Load and validate schema -----------------------------------------
    print("\n1. Loading dataset...")
    try:
        dataset = load_evaluation_dataset(DATASET_PATH)
        print(f"   ✓ Loaded {len(dataset.questions)} questions (schema v{dataset.schema_version})")
    except Exception as exc:
        print(f"   ✗ Failed to load: {exc}")
        sys.exit(1)

    # ---- 2. Compute statistics -----------------------------------------------
    print("\n2. Dataset statistics:")
    stats = compute_dataset_stats(dataset)
    for key, value in sorted(stats.items()):
        print(f"   {key}: {value}")

    # ---- 3. Check corpus coverage --------------------------------------------
    print("\n3. Corpus coverage:")
    corpus = load_corpus_text()
    if not corpus:
        print("   ⚠ Corpus directory empty or missing — skipping coverage checks")
    else:
        print(f"   Corpus contains {len(corpus)} documents")
        coverage_warnings = validate_corpus_coverage(dataset, set(corpus.keys()))
        if coverage_warnings:
            print(f"   ⚠ {len(coverage_warnings)} coverage warnings:")
            for warning in coverage_warnings[:5]:
                print(f"     - {warning}")
            if len(coverage_warnings) > 5:
                print(f"     ... and {len(coverage_warnings) - 5} more")
        else:
            print("   ✓ All expected_sources exist in corpus")

        # ---- 4. Check evidence spans -----------------------------------------
        print("\n4. Evidence span verification:")
        evidence_warnings = check_evidence_in_corpus(dataset, corpus)
        if evidence_warnings:
            print(f"   ⚠ {len(evidence_warnings)} evidence warnings:")
            for warning in evidence_warnings[:5]:
                print(f"     - {warning}")
            if len(evidence_warnings) > 5:
                print(f"     ... and {len(evidence_warnings) - 5} more")
        else:
            print("   ✓ All evidence spans found in corpus")

    # ---- 5. Check category balance -------------------------------------------
    print("\n5. Category balance:")
    unbalanced = []
    for category in [
        "simple_factual",
        "multi_hop",
        "ambiguous",
        "unanswerable",
        "temporal_version",
        "metadata_filter",
    ]:
        pct = stats.get(f"pct_{category}", 0)
        if pct < MIN_CATEGORY_PCT:
            unbalanced.append((category, pct))
    if unbalanced:
        print("   ⚠ Underrepresented categories:")
        for category, pct in unbalanced:
            print(f"     - {category}: {pct}% (minimum {MIN_CATEGORY_PCT}%)")
    else:
        print("   ✓ All categories have ≥5% representation")

    # ---- 6. Check unanswerable format ----------------------------------------
    print("\n6. Unanswerable question format:")
    unanswerable_warnings = check_unanswerable_format(dataset)
    if unanswerable_warnings:
        print(f"   ⚠ {len(unanswerable_warnings)} format warnings:")
        for warning in unanswerable_warnings:
            print(f"     - {warning}")
    else:
        print("   ✓ All unanswerable questions formatted correctly")

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 80)
    total_warnings = (
        len(coverage_warnings if corpus else [])
        + len(evidence_warnings if corpus else [])
        + len(unbalanced)
        + len(unanswerable_warnings)
    )
    if total_warnings == 0:
        print("✓ Dataset validation PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"⚠ Dataset validation completed with {total_warnings} warning(s)")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()