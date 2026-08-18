"""Grounding score computation + response policy (Phase 13).

Grounding score (D-071):
    score = (supported + 0.5 * partial) / total_claims   [0.0 .. 1.0]
This is the OBJECTIVE faithfulness metric — distinct from the LLM's
self-reported confidence (D-062). Both are reported; neither is hidden.

Response policy (D-072): configurable thresholds drive one of four outcomes:
1. ACCEPT:        grounding_score >= threshold → return as-is
2. REGENERATE:    score < threshold, attempts < max → regenerate with stricter prompt
3. RETRIEVE_MORE: expand context window and regenerate (once)
4. UNCERTAINTY:   return the answer flagged with low confidence + explicit notice

The policy is honest: it never claims hallucinations are "eliminated".
It reports what was detected and what action was taken.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.core.config import GroundingSettings
from app.core.logging import get_logger
from app.generation.domain import CitationValidation, ClaimVerification
from app.generation.matching import ClaimStatus

logger = get_logger(__name__)

_PARTIAL_WEIGHT = 0.5


class PolicyAction(StrEnum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    RETRIEVE_MORE = "retrieve_more"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class GroundingScore:
    score: float                       # [0.0 .. 1.0]
    supported: int
    partial: int
    unsupported: int
    total: int
    judge_adjudications: int = 0       # claims sent to the LLM judge
    judge_failures: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str
    regeneration_attempt: int          # how many times we've regenerated so far
    context_expanded: bool = False


def compute_grounding_score(validation: CitationValidation) -> GroundingScore:
    total = validation.claims_total
    if total == 0:
        return GroundingScore(
            score=1.0 if validation.verdict.value == "clean" else 0.0,
            supported=0, partial=0, unsupported=0, total=0,
        )
    score = (validation.claims_supported + _PARTIAL_WEIGHT * validation.claims_partial) / total
    return GroundingScore(
        score=round(score, 3),
        supported=validation.claims_supported,
        partial=validation.claims_partial,
        unsupported=validation.claims_unsupported,
        total=total,
    )


def decide_action(
    grounding: GroundingScore,
    settings: GroundingSettings,
    *,
    regeneration_attempt: int,
    context_already_expanded: bool,
) -> PolicyDecision:
    if grounding.total == 0:
        return PolicyDecision(PolicyAction.ACCEPT, "no claims to evaluate", regeneration_attempt)

    if grounding.score >= settings.poor_grounding_threshold:
        return PolicyDecision(PolicyAction.ACCEPT, "grounding adequate", regeneration_attempt)

    if (
        settings.retrieve_more_context_enabled
        and not context_already_expanded
        and regeneration_attempt == 0
    ):
        return PolicyDecision(
            PolicyAction.RETRIEVE_MORE,
            "grounding below threshold; expanding context",
            regeneration_attempt,
            context_expanded=True,
        )

    if regeneration_attempt < settings.max_regeneration_attempts:
        return PolicyDecision(
            PolicyAction.REGENERATE,
            "grounding below threshold; regenerating with stricter prompt",
            regeneration_attempt,
        )

    return PolicyDecision(
        PolicyAction.UNCERTAINTY,
        f"grounding {grounding.score:.2f} below threshold {settings.poor_grounding_threshold:.2f}; "
        f"exhausted {regeneration_attempt} regeneration attempt(s)",
        regeneration_attempt,
    )