"""ClaimAdjudicator — runs the LLM judge on the partial band (D-069).

Strategy:
- Supported claims: never re-judged (cost savings, determinism preserved)
- Partial claims: always sent to the judge (this is the reserved band)
- Unsupported claims with markers: sent to the judge ONLY when
  `judge_unsupported` is enabled (config; default True) — some "unsupported"
  claims are genuinely unsupported, others are weak matches the deterministic
  matcher missed.
- Unmarked claims: never judged (no evidence to adjudicate against)

Updates ClaimVerification in place, preserving the deterministic matcher's
best_score as `deterministic_score` for observability.
"""

from __future__ import annotations

from app.core.config import GroundingSettings
from app.core.logging import get_logger
from app.generation.context import ContextPack
from app.generation.domain import ClaimVerification
from app.generation.judge import ClaimJudge, status_from_verdict
from app.generation.matching import ClaimStatus

logger = get_logger(__name__)


class ClaimAdjudicator:
    def __init__(self, judge: ClaimJudge, settings: GroundingSettings) -> None:
        self._judge = judge
        self._settings = settings

    async def adjudicate(
        self,
        claims: list[ClaimVerification],
        pack: ContextPack,
    ) -> tuple[list[ClaimVerification], int, int]:
        """Returns (updated_claims, adjudications_run, judge_failures)."""
        adjudications = 0
        failures = 0
        updated: list[ClaimVerification] = []

        for claim in claims:
            if claim.unmarked:
                updated.append(claim)
                continue

            status = ClaimStatus(claim.status)
            should_judge = (
                status == ClaimStatus.PARTIAL
                or (status == ClaimStatus.UNSUPPORTED and self._settings.judge_unsupported and claim.markers)
            )
            if not should_judge:
                updated.append(claim)
                continue

            # Build the best evidence from the passages this claim cited.
            evidence_parts: list[str] = []
            for marker in claim.markers:
                passage = pack.passage_by_number(marker)
                if passage is not None:
                    evidence_parts.append(passage.chunk.content)
            if not evidence_parts:
                updated.append(claim)
                continue

            evidence = "\n\n".join(evidence_parts)
            outcome = await self._judge.adjudicate(claim.claim, evidence)
            adjudications += 1
            if outcome.failed:
                failures += 1
                updated.append(claim)  # preserve Phase 12 verdict
                continue

            new_status = status_from_verdict(outcome.verdict)
            updated.append(
                ClaimVerification(
                    claim=claim.claim,
                    markers=claim.markers,
                    status=new_status.value,
                    best_score=outcome.reason and 1.0 if new_status == ClaimStatus.SUPPORTED else claim.best_score,
                    unmarked=False,
                )
            )
            logger.info(
                "claim_adjudicated",
                verdict=outcome.verdict.value,
                latency_ms=outcome.latency_ms,
            )

        return updated, adjudications, failures