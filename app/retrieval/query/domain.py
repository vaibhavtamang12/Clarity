"""Query transformation domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TransformType(StrEnum):
    NONE = "none"
    REWRITE = "rewrite"                     # conversational follow-up → standalone
    EXPANSION = "expansion"                 # sparse query + key terms
    DECOMPOSITION = "decomposition"         # complex question → sub-questions
    MULTI = "multi"                         # more than one transform applied


@dataclass(frozen=True)
class ConversationTurn:
    role: str        # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class RoutingDecision:
    needs_conversational_rewrite: bool
    needs_expansion: bool
    needs_decomposition: bool
    reasons: tuple[str, ...] = ()

    @property
    def needs_any(self) -> bool:
        return self.needs_conversational_rewrite or self.needs_expansion or self.needs_decomposition


@dataclass(frozen=True)
class TransformedQuery:
    original: str
    retrieval_query: str                     # what retrieval actually executes
    transform_type: TransformType
    rewritten: str | None = None
    expansion_terms: tuple[str, ...] = ()
    sub_queries: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    degraded: bool = False                   # some transform failed → fell back
    degraded_reason: str | None = None
    llm_latency_ms: float = 0.0
    token_usage: int = 0