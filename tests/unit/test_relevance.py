"""Tests for evidence-based relevance adjudication."""

from __future__ import annotations

from app.evaluation.relevance import (
    evidence_contained,
    is_relevant,
    relevant_chunk_indices,
    token_overlap_ratio,
)

CHUNK = "Enterprise customers have a 30-day refund window from the invoice date."


def test_exact_containment() -> None:
    assert evidence_contained(CHUNK, "Enterprise customers have a 30-day refund window from the invoice date.")


def test_containment_is_normalized() -> None:
    assert evidence_contained(CHUNK, "  enterprise   customers have a 30-day REFUND window from the invoice date. ")


def test_unrelated_evidence_not_contained() -> None:
    assert not evidence_contained(CHUNK, "Passwords must be at least 14 characters long.")


def test_overlap_ratio() -> None:
    ratio = token_overlap_ratio(CHUNK, "Enterprise customers have a 30-day refund window")
    assert ratio > 0.9
    assert token_overlap_ratio(CHUNK, "completely different text about dragons") < 0.3


def test_is_relevant_threshold() -> None:
    # most evidence tokens present even without exact containment
    assert is_relevant(CHUNK, "Enterprise customers have a 30-day refund window from invoice date")
    assert not is_relevant(CHUNK, "Quarterly access reviews are performed by system owners")


def test_relevant_chunk_indices_union() -> None:
    chunks = [CHUNK, "Passwords must be at least 14 characters long.", "Noise only."]
    indices = relevant_chunk_indices(
        chunks,
        [
            "Enterprise customers have a 30-day refund window from the invoice date.",
            "Passwords must be at least 14 characters long.",
        ],
    )
    assert indices == {0, 1}