"""CitationEngine — citation validation across three integrity dimensions.

1. ANSWER CLAIMS: every marked sentence must be supported by at least one of
   the passages it cites. Unmarked factual sentences are flagged as such.
2. DECLARED CITATIONS: the LLM's per-citation claim must match the cited chunk.
3. STRUCTURAL HYGIENE:
   - dangling markers: [n] in the answer that never resolved (hallucinated refs)
   - unused citations: resolved but never referenced — the classic
     "append random retrieved documents" anti-pattern the spec forbids.

Policy decisions (D-065, D-067):
- When the answer uses markers, unreferenced citations are PRUNED.
- When it uses none, all citations are kept but the result is flagged.
- Dangling markers are flagged, never silently rewritten (Rule 10 honesty).

Matching is deterministic (D-066); the entailment/LLM judge lands in Phase 13.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.generation.citations import CitationResolution
from app.generation.claims import all_markers, extract_claims
from app.generation.context import ContextPack
from app.generation.domain import (
    Citation,
    CitationCheck,
    CitationValidation,
    CitationVerdict,
    ClaimVerification,
)
from app.generation.matching import ClaimStatus, best_match_against, match_claim_to_chunk

logger = get_logger(__name__)

_POOR_SUPPORT_THRESHOLD = 0.5


class CitationEngine:
    def validate(
        self,
        answer: str,
        resolution: CitationResolution,
        pack: ContextPack,
    ) -> CitationValidation:
        resolved = resolution.citations
        content_by_passage: dict[int, str] = {}
        for citation in resolved:
            passage = pack.passage_by_number(citation.passage)
            if passage is not None:
                content_by_passage[citation.passage] = passage.chunk.content

        # ---- 1. answer claims -------------------------------------------------
        claims = extract_claims(answer)
        referenced_markers = all_markers(answer)
        resolved_passages = {c.passage for c in resolved}

        verifications: list[ClaimVerification] = []
        supported = partial = unsupported = 0
        for claim in claims:
            if not claim.markers:
                verifications.append(
                    ClaimVerification(
                        claim=claim.text, status=ClaimStatus.UNSUPPORTED.value,
                        best_score=0.0, unmarked=True,
                    )
                )
                unsupported += 1
                continue
            candidate_contents = [
                content_by_passage[m] for m in claim.markers if m in content_by_passage
            ]
            if not candidate_contents:
                # all of this claim's markers are dangling
                verifications.append(
                    ClaimVerification(
                        claim=claim.text, markers=list(claim.markers),
                        status=ClaimStatus.UNSUPPORTED.value, best_score=0.0,
                    )
                )
                unsupported += 1
                continue
            match = best_match_against(claim.text, candidate_contents)
            verifications.append(
                ClaimVerification(
                    claim=claim.text, markers=list(claim.markers),
                    status=match.status.value, best_score=round(match.best_score, 3),
                )
            )
            if match.status == ClaimStatus.SUPPORTED:
                supported += 1
            elif match.status == ClaimStatus.PARTIAL:
                partial += 1
            else:
                unsupported += 1

        claims_total = len(claims)
        support_rate = supported / claims_total if claims_total else 1.0

        # ---- 2. declared citation claims ---------------------------------------
        checks: list[CitationCheck] = []
        verified_declared = 0
        declared_total = 0
        for citation in resolved:
            declared = citation.claim.strip()
            if not declared:
                continue
            declared_total += 1
            content = content_by_passage.get(citation.passage, "")
            match = match_claim_to_chunk(declared, content)
            checks.append(
                CitationCheck(
                    passage=citation.passage, chunk_id=citation.chunk_id,
                    declared_claim=declared[:300], status=match.status.value,
                    score=round(match.best_score, 3),
                )
            )
            if match.status == ClaimStatus.SUPPORTED:
                verified_declared += 1
        citation_correctness = verified_declared / declared_total if declared_total else 1.0

        # ---- 3. structural hygiene ----------------------------------------------
        dangling = sorted(referenced_markers - resolved_passages)
        markers_present = bool(referenced_markers)
        if markers_present:
            final_citations = [c for c in resolved if c.passage in referenced_markers]
            unused = sorted(resolved_passages - referenced_markers)
        else:
            final_citations = list(resolved)   # kept, but flagged as unverified
            unused = []

        # ---- verdict --------------------------------------------------------------
        issues: list[str] = []
        if unsupported:
            issues.append(f"{unsupported} unsupported claim(s)")
        if partial:
            issues.append(f"{partial} partially supported claim(s)")
        if dangling:
            issues.append(f"dangling markers {dangling}")
        if unused:
            issues.append(f"{len(unused)} unused citation(s) pruned")
        if declared_total and verified_declared < declared_total:
            issues.append(f"{declared_total - verified_declared} declared citation(s) unverified")
        if not markers_present and resolved:
            issues.append("answer contains no citation markers — citations unverified")

        if claims_total and support_rate < _POOR_SUPPORT_THRESHOLD:
            verdict = CitationVerdict.POOR
        elif issues:
            verdict = CitationVerdict.FLAGGED
        else:
            verdict = CitationVerdict.CLEAN

        validation = CitationValidation(
            verdict=verdict,
            final_citations=final_citations,
            claims=verifications,
            citation_checks=checks,
            claims_total=claims_total,
            claims_supported=supported,
            claims_partial=partial,
            claims_unsupported=unsupported,
            claim_support_rate=round(support_rate, 3),
            dangling_markers=dangling,
            unused_citations=unused,
            markers_present=markers_present,
            citation_correctness=round(citation_correctness, 3),
            summary="; ".join(issues) if issues else "all claims supported; citations verified",
        )
        logger.info(
            "citation_validation_completed",
            verdict=verdict.value,
            claims=claims_total,
            support_rate=validation.claim_support_rate,
            dangling=len(dangling),
        )
        return validation