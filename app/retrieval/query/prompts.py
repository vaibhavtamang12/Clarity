"""Versioned prompt templates for query transformation (ADR-005).

Prompt identity (PROMPT_VERSION) travels with every log/metric so quality
regressions can be attributed to prompt changes. LLM output is UNTRUSTED:
every consumer parses, bounds, and sanitizes before use (Rule 11).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from app.retrieval.query.domain import ConversationTurn

PROMPT_VERSION = "query-transform/1"

REWRITE_SYSTEM = (
    "You are a search-query rewriter for a document knowledge base.\n"
    "Your ONLY task: rewrite the user's follow-up question into ONE standalone "
    "search question that fully captures its intent, using the conversation context.\n"
    "Rules:\n"
    "- Never answer the question.\n"
    "- Never invent facts absent from the conversation.\n"
    "- Output exactly one line: the rewritten question. No quotes, no explanations."
)

REWRITE_USER_TEMPLATE = """Conversation:
[user] What is the refund policy?
[assistant] Standard plans have a 14-day refund window.
Follow-up: What about enterprise customers?

Standalone question: What is the refund window for enterprise customers?

Conversation:
{conversation}
Follow-up: {query}

Standalone question:"""

EXPANSION_SYSTEM = (
    "You expand sparse search queries for a document knowledge base.\n"
    "Produce 3-8 precise search terms or short phrases that a lexical or dense "
    "retriever should also match for the given query.\n"
    "Output STRICT JSON only: {\"terms\": [\"...\", \"...\"]}"
)

EXPANSION_USER_TEMPLATE = "Query: {query}\nTerms JSON:"

DECOMPOSE_SYSTEM = (
    "You split complex multi-part questions into simpler standalone sub-questions "
    "for a document knowledge base.\n"
    "Produce 2-4 sub-questions. Each must be answerable independently.\n"
    "Output STRICT JSON only: {\"sub_questions\": [\"...\", \"...\"]}"
)

DECOMPOSE_USER_TEMPLATE = "Question: {query}\nSub-questions JSON:"


def format_conversation(turns: Sequence[ConversationTurn]) -> str:
    return "\n".join(f"[{turn.role}] {turn.content}" for turn in turns)


def parse_json_field(raw: str, field: str) -> list[str]:
    """Bounded, sanitized JSON extraction. Raises ValueError on any deviation —
    callers treat that as 'transform unavailable' and fall back (D-054)."""
    text = raw.strip()
    if text.startswith("```"):  # tolerate fenced answers without trusting them
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    values = data.get(field)
    if not isinstance(values, list):
        raise ValueError(f"expected '{field}' list")
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        raise ValueError("empty result list")
    return cleaned