"""RAGGenerator — structured generation with schema-repair retry (ADR-005).

Contract:
- json_output mode requested from the provider when supported
- on parse/schema failure: one repair attempt (bad output echoed back with a
  correction instruction)
- on provider failure or exhausted repairs: returns outcome.output=None —
  the pipeline decides the fallback (retrieval-only), the user never sees
  a raw model error and never sees a fabricated answer (D-059)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import GenerationSettings
from app.core.logging import get_logger
from app.generation.context import ContextPack
from app.generation.domain import LLMGenerationOutput
from app.generation.prompts import (
    REPAIR_INSTRUCTION,
    SYSTEM_PROMPT,
    build_user_message,
    parse_generation_output,
)
from app.llm.base import ChatMessage, LLMProvider, LLMRequest, TokenUsage

logger = get_logger(__name__)


@dataclass
class GenerationOutcome:
    output: LLMGenerationOutput | None
    attempts: int
    latency_ms: float
    usage: TokenUsage
    error: str | None = None


class RAGGenerator:
    def __init__(self, provider: LLMProvider, settings: GenerationSettings) -> None:
        self._provider = provider
        self._settings = settings

    async def generate(self, question: str, context: ContextPack) -> GenerationOutcome:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_message(question, context.text)),
        ]
        usage = TokenUsage()
        max_attempts = 1 + max(0, self._settings.repair_attempts)
        last_error = "unknown"

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._provider.generate(
                    LLMRequest(
                        messages=messages,
                        temperature=self._settings.temperature,
                        max_tokens=self._settings.max_answer_tokens,
                        json_output=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — typed fallback upstream
                logger.warning("generation_provider_failed", error=type(exc).__name__)
                return GenerationOutcome(
                    output=None, attempts=attempt, latency_ms=0.0, usage=usage,
                    error=f"provider:{type(exc).__name__}",
                )

            usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens + response.usage.prompt_tokens,
                completion_tokens=usage.completion_tokens + response.usage.completion_tokens,
            )
            try:
                output = parse_generation_output(response.content)
                logger.info(
                    "generation_completed",
                    attempt=attempt,
                    latency_ms=response.latency_ms,
                    citations=len(output.citations),
                )
                return GenerationOutcome(
                    output=output, attempts=attempt,
                    latency_ms=response.latency_ms, usage=usage,
                )
            except ValueError as exc:
                last_error = str(exc)
                logger.warning("generation_schema_invalid", attempt=attempt, error=last_error)
                if attempt < max_attempts:
                    messages.append(ChatMessage(role="assistant", content=response.content))
                    messages.append(ChatMessage(role="user", content=REPAIR_INSTRUCTION))

        return GenerationOutcome(
            output=None, attempts=max_attempts, latency_ms=0.0, usage=usage,
            error=f"schema:{last_error[:200]}",
        )