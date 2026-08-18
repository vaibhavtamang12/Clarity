"""Scripted mock provider with streaming support."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from app.llm.base import LLMRequest, LLMResponse, StreamEvent, TokenUsage


class ScriptedMockProvider:
    """Returns scripted responses in order (last one repeats). Supports streaming."""

    name = "mock"

    def __init__(self, responses: str | Sequence[str], *, stream_delay: float = 0.0) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        if not self._responses:
            raise ValueError("ScriptedMockProvider needs at least one response")
        self.calls: list[LLMRequest] = []
        self._stream_delay = stream_delay

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

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream the scripted response token-by-token (simulated)."""
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        content = self._responses[index]
        
        # Simulate token-by-token streaming (split on spaces)
        tokens = content.split(" ")
        for token in tokens:
            if self._stream_delay > 0:
                await asyncio.sleep(self._stream_delay)
            yield StreamEvent(token=token + " ", model="mock-model")
        
        prompt_chars = sum(len(m.content) for m in request.messages)
        yield StreamEvent(
            done=True,
            usage=TokenUsage(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=max(1, len(content) // 4),
            ),
        )