"""Rule-based query router (decision D-053).

The router is the cheapest possible gate: deterministic string/token rules,
sub-millisecond, zero LLM cost. An LLM router would add latency to EVERY
query to decide whether to spend latency — a budget violation, not an
engineering choice (Rule 6; trade-off documented per Rule 14).

Signals:
- conversational: follow-up markers (pronouns/demonstratives/"what about"),
  or a short query when conversation history exists
- expansion: very sparse standalone queries
- decomposition: multi-question or multi-part connectives in long queries
"""

from __future__ import annotations

from app.core.config import QueryTransformSettings
from app.retrieval.bm25 import tokenize
from app.retrieval.query.domain import RoutingDecision

_FOLLOWUP_MARKERS = {
    "it", "its", "they", "their", "them", "that", "this", "those", "these",
    "he", "she", "his", "her", "there", "then",
}
_FOLLOWUP_PHRASES = (
    "what about", "how about", "and what", "also", "as well",
    "compare", "versus", "vs", "same for", "in that case",
)
_MULTIPART_PHRASES = (
    "and also", "as well as", "first", "then", "compare", "difference between",
    "step by step", "on one hand",
)

_SHORT_FOLLOWUP_TOKENS = 8


class QueryRouter:
    def __init__(self, settings: QueryTransformSettings) -> None:
        self._settings = settings

    def route(self, query: str, *, conversation_turns: int = 0) -> RoutingDecision:
        if not self._settings.enabled:
            return RoutingDecision(False, False, False, ("transform disabled",))

        tokens = tokenize(query)
        lowered = query.lower()
        reasons: list[str] = []

        # ---- conversational rewriting ----------------------------------------
        needs_rewrite = False
        if conversation_turns > 0:
            has_marker = any(t in _FOLLOWUP_MARKERS for t in tokens[:6]) or any(
                p in lowered for p in _FOLLOWUP_PHRASES
            )
            is_short_followup = len(tokens) <= _SHORT_FOLLOWUP_TOKENS
            if has_marker:
                needs_rewrite = True
                reasons.append("follow-up markers present")
            elif is_short_followup:
                needs_rewrite = True
                reasons.append("short follow-up with active conversation")

        # ---- expansion --------------------------------------------------------
        needs_expansion = (
            conversation_turns == 0
            and 0 < len(tokens) <= self._settings.expansion_min_query_tokens
        )
        if needs_expansion:
            reasons.append("sparse standalone query")

        # ---- decomposition ----------------------------------------------------
        needs_decomposition = (
            len(tokens) >= self._settings.decompose_min_tokens
            and (
                query.count("?") > 1
                or any(p in lowered for p in _MULTIPART_PHRASES)
            )
        )
        if needs_decomposition:
            reasons.append("multi-part or multi-question query")

        return RoutingDecision(
            needs_conversational_rewrite=needs_rewrite,
            needs_expansion=needs_expansion,
            needs_decomposition=needs_decomposition,
            reasons=tuple(reasons),
        )