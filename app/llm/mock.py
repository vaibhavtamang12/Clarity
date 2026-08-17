"""Scripted mock provider for deterministic tests (no network, no secrets)."""

from __future__ import annotations

from collections.abc import Sequence

from app.llm.base import LLMRequest, LLMResponse, TokenUsage


class ScriptedMockProvider:
    """Returns scripted responses in order (last one repeats). Records every
    request so tests can assert WHAT was prompted and HOW OFTEN."""

    name = "mock"

    def __init__(self, responses: str | Sequence[str]) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        if not self._responses:
            raise ValueError("ScriptedMockProvider needs at least one response")
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        content = self._responses[index]
        prompt_chars = sum(len(m.content) for m in request.messages)
        return LLMResponse(
            content=content,
            model="mock-model",
            usage=TokenUsage(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=max(1, len(content) // 4),
            ),
            latency_ms=0.1,
        )