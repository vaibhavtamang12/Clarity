"""LLM provider abstraction (ADR-005, partially realized — D-055).

The protocol every provider implements. Request/response are Pydantic models
so token accounting and validation are structural, not aspirational.
Phase 10 consumes generate(); Phase 17 adds stream() to the same protocol.
"""

from __future__ import annotations

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


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...