# app/conversation/history.py
"""History selection — recency + embedding relevance (D-074, unchanged policy)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.conversation.domain import ConversationTurn
from app.embeddings.service import EmbeddingService


@dataclass(frozen=True)
class HistorySelectionConfig:
    max_turns: int = 5
    recency_weight: float = 0.4
    relevance_weight: float = 0.6
    min_score_threshold: float = 0.3


class HistorySelector:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: HistorySelectionConfig | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._config = config or HistorySelectionConfig()

    async def select_relevant_turns(
        self, query: str, turns: Sequence[ConversationTurn]
    ) -> list[ConversationTurn]:
        if not turns:
            return []
        scored: list[tuple[ConversationTurn, float]] = []
        for turn in turns:
            score = await self._score_turn(query, turn, len(turns))
            if score >= self._config.min_score_threshold:
                scored.append((turn, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = [turn for turn, _ in scored[: self._config.max_turns]]
        selected.sort(key=lambda turn: turn.turn_index)
        return selected

    async def _score_turn(self, query: str, turn: ConversationTurn, total: int) -> float:
        recency = (turn.turn_index + 1) / total
        try:
            query_vec = await self._embedding_service.embed_query(query)
            turn_vec = await self._embedding_service.embed_query(
                f"{turn.user_message} {turn.assistant_message}"
            )
            dot = sum(a * b for a, b in zip(query_vec, turn_vec))
            norm_q = sum(a * a for a in query_vec) ** 0.5
            norm_t = sum(b * b for b in turn_vec) ** 0.5
            relevance = dot / (norm_q * norm_t) if norm_q and norm_t else 0.0
        except Exception:  # noqa: BLE001 — relevance degrades to recency-only
            relevance = 0.0
        return self._config.recency_weight * recency + self._config.relevance_weight * relevance