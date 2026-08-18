"""LLM provider abstraction with streaming support (Phase 17)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    json_output: bool = False
    stop: list[str] | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    content: str
    model: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0


class StreamEvent(BaseModel):
    """A single streaming event from the LLM provider."""
    
    token: str | None = None              # incremental token
    done: bool = False                    # True when generation complete
    usage: TokenUsage | None = None       # final usage (only when done=True)
    model: str | None = None
    error: str | None = None              # error message if generation failed


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream tokens incrementally. Yields StreamEvent with token content,
        then a final event with done=True and usage stats."""
        ...