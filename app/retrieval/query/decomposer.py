# app/retrieval/query/decomposer.py
"""Question decomposition for complex multi-part queries.

Phase 10 produces validated sub-questions; Phase 11's pipeline decides how
to fan them out over retrieval. Output is bounded to 2-4 sub-questions.
"""

from __future__ import annotations

from app.core.config import QueryTransformSettings
from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider, LLMRequest
from app.retrieval.query.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_USER_TEMPLATE, parse_json_field

logger = get_logger(__name__)

_MIN_SUBQUESTIONS = 2
_MAX_SUBQUESTIONS = 4
_MAX_SUBQUESTION_CHARS = 200


class QueryDecomposer:
    def __init__(self, provider: LLMProvider, settings: QueryTransformSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def decompose(self, query: str) -> tuple[tuple[str, ...], float]:
        request = LLMRequest(
            messages=[
                ChatMessage(role="system", content=DECOMPOSE_SYSTEM),
                ChatMessage(role="user", content=DECOMPOSE_USER_TEMPLATE.format(query=query)),
            ],
            temperature=self._settings.temperature,
            max_tokens=256,
            json_output=True,
        )
        try:
            response = await self._provider.generate(request)
            sub_questions = parse_json_field(response.content, "sub_questions")
        except Exception as exc:  # noqa: BLE001 — fail-safe by contract
            logger.warning("decomposition_failed", error=type(exc).__name__)
            return (), 0.0

        cleaned = [q.strip()[:_MAX_SUBQUESTION_CHARS] for q in sub_questions if q.strip()]
        if len(cleaned) < _MIN_SUBQUESTIONS:
            return (), response.latency_ms
        return tuple(cleaned[:_MAX_SUBQUESTIONS]), response.latency_ms