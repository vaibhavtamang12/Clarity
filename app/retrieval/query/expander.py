# app/retrieval/query/expander.py
"""Query expansion for sparse standalone queries (LLM-proposed terms)."""

from __future__ import annotations

from app.core.config import QueryTransformSettings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider, LLMRequest
from app.retrieval.query.prompts import EXPANSION_SYSTEM, EXPANSION_USER_TEMPLATE, parse_json_field

logger = get_logger(__name__)

_MAX_TERM_CHARS = 40


class QueryExpander:
    def __init__(self, provider: LLMProvider, settings: QueryTransformSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def expand(self, query: str) -> tuple[tuple[str, ...], float]:
        request = LLMRequest(
            messages=[
                ChatMessage(role="system", content=EXPANSION_SYSTEM),
                ChatMessage(role="user", content=EXPANSION_USER_TEMPLATE.format(query=query)),
            ],
            temperature=self._settings.temperature,
            max_tokens=128,
            json_output=True,
        )
        try:
            response = await self._provider.generate(request)
            terms = parse_json_field(response.content, "terms")
        except Exception as exc:  # noqa: BLE001 — fail-safe by contract
            logger.warning("expansion_failed", error=type(exc).__name__)
            return (), 0.0

        seen: set[str] = set()
        cleaned: list[str] = []
        for term in terms:
            t = term.strip().strip('"')[:_MAX_TERM_CHARS]
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                cleaned.append(t)
        return tuple(cleaned[: self._settings.expansion_max_terms]), response.latency_ms