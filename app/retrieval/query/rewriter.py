# app/retrieval/query/rewriter.py
"""Conversational query rewriting — follow-up → standalone question.

Fail-safe contract (D-054): any LLM failure returns None; the caller keeps
the original query. A rewrite is accepted only if it is non-empty, bounded,
and different from the original.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import QueryTransformSettings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider, LLMRequest
from app.retrieval.query.domain import ConversationTurn
from app.retrieval.query.prompts import REWRITE_SYSTEM, REWRITE_USER_TEMPLATE, format_conversation

logger = get_logger(__name__)


class ConversationalRewriter:
    def __init__(self, provider: LLMProvider, settings: QueryTransformSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def rewrite(
        self, query: str, history: Sequence[ConversationTurn]
    ) -> tuple[str | None, float]:
        """Returns (rewritten_query_or_None, llm_latency_ms)."""
        window = list(history)[-self._settings.history_window :]
        request = LLMRequest(
            messages=[
                ChatMessage(role="system", content=REWRITE_SYSTEM),
                ChatMessage(
                    role="user",
                    content=REWRITE_USER_TEMPLATE.format(
                        conversation=format_conversation(window), query=query
                    ),
                ),
            ],
            temperature=self._settings.temperature,
            max_tokens=self._settings.rewrite_max_output_tokens,
        )
        try:
            response = await self._provider.generate(request)
        except Exception as exc:  # noqa: BLE001 — fail-safe by contract
            logger.warning("rewrite_failed", error=type(exc).__name__)
            return None, 0.0

        candidate = response.content.strip().splitlines()[0].strip() if response.content.strip() else ""
        candidate = candidate.strip('"').strip()[: self._settings.max_output_chars]
        if not candidate or candidate.lower() == query.lower():
            return None, response.latency_ms
        logger.info("query_rewritten", latency_ms=response.latency_ms)
        return candidate, response.latency_ms