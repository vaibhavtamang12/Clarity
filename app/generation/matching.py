"""Claim ↔ chunk evidence matching (decision D-063).

Reuses the evaluation harness's relevance adjudication — containment or
≥0.7 token overlap — so citation validation and retrieval evaluation can
never drift apart methodologically. Direction: claim tokens ⊆ chunk tokens.

A PARTIAL band (≥0.4) separates "clearly unsupported" from "weakly
supported" — Phase 13's judge will adjudicate that band with entailment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.evaluation.relevance import DEFAULT_MIN_OVERLAP, evidence_contained, token_overlap_ratio

PARTIAL_MIN_OVERLAP = 0.4


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class MatchResult:
    status: ClaimStatus
    best_score: float
    contained: bool


def match_claim_to_chunk(claim_text: str, chunk_content: str) -> MatchResult:
    if not claim_text.strip() or not chunk_content.strip():
        return MatchResult(status=ClaimStatus.UNSUPPORTED, best_score=0.0, contained=False)
    if evidence_contained(chunk_content, claim_text):
        return MatchResult(status=ClaimStatus.SUPPORTED, best_score=1.0, contained=True)
    score = token_overlap_ratio(chunk_content, claim_text)
    if score >= DEFAULT_MIN_OVERLAP:
        return MatchResult(status=ClaimStatus.SUPPORTED, best_score=score, contained=False)
    if score >= PARTIAL_MIN_OVERLAP:
        return MatchResult(status=ClaimStatus.PARTIAL, best_score=score, contained=False)
    return MatchResult(status=ClaimStatus.UNSUPPORTED, best_score=score, contained=False)


def best_match_against(claim_text: str, chunk_contents: Sequence[str]) -> MatchResult:
    best = MatchResult(status=ClaimStatus.UNSUPPORTED, best_score=0.0, contained=False)
    for content in chunk_contents:
        result = match_claim_to_chunk(claim_text, content)
        if result.status.value > best.status.value or (
            result.status == best.status and result.best_score > best.best_score
        ):
            best = result
    return best