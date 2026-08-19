"""Token usage and cost tracking (Phase 23).

Tracks token usage (prompt + completion) per query and estimates cost based
on configurable model pricing. This feeds the cost analysis in the benchmark
report and helps make informed production decisions (Rule 2: measure, don't guess).

Pricing is configurable via settings, so it works with any provider. The
default pricing table covers common OpenAI models but can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Default pricing table: cost per 1,000 tokens (USD)
# Override via settings or environment variables for production accuracy
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "llama3.1:8b": {"prompt": 0.0, "completion": 0.0},  # local = free
    "default": {"prompt": 0.001, "completion": 0.002},  # fallback
}


@dataclass(frozen=True)
class TokenUsageStats:
    """Token usage statistics for a benchmark run."""

    total_queries: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_prompt_tokens_per_query: float
    avg_completion_tokens_per_query: float
    avg_tokens_per_query: float
    total_cost_usd: float
    avg_cost_per_query_usd: float


class CostTracker:
    """Tracks token usage and estimates cost."""

    def __init__(self, pricing: dict[str, dict[str, float]] | None = None) -> None:
        self._pricing = pricing or DEFAULT_PRICING
        self._prompt_tokens: list[int] = []
        self._completion_tokens: list[int] = []
        self._model_name: str = "default"

    def set_model(self, model_name: str) -> None:
        """Set the model name for pricing lookup."""
        self._model_name = model_name

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for a single query."""
        self._prompt_tokens.append(prompt_tokens)
        self._completion_tokens.append(completion_tokens)

    def _get_pricing(self) -> dict[str, float]:
        """Get pricing for the current model, falling back to default."""
        return self._pricing.get(self._model_name, self._pricing.get("default", {"prompt": 0, "completion": 0}))

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute cost for a single query."""
        pricing = self._get_pricing()
        prompt_cost = (prompt_tokens / 1000) * pricing.get("prompt", 0)
        completion_cost = (completion_tokens / 1000) * pricing.get("completion", 0)
        return prompt_cost + completion_cost

    def get_stats(self) -> TokenUsageStats:
        """Compute aggregate token usage and cost statistics."""
        total_queries = len(self._prompt_tokens)
        if total_queries == 0:
            return TokenUsageStats(
                total_queries=0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                avg_prompt_tokens_per_query=0.0,
                avg_completion_tokens_per_query=0.0,
                avg_tokens_per_query=0.0,
                total_cost_usd=0.0,
                avg_cost_per_query_usd=0.0,
            )

        total_prompt = sum(self._prompt_tokens)
        total_completion = sum(self._completion_tokens)
        total_tokens = total_prompt + total_completion
        total_cost = sum(
            self._compute_cost(p, c)
            for p, c in zip(self._prompt_tokens, self._completion_tokens)
        )

        return TokenUsageStats(
            total_queries=total_queries,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            avg_prompt_tokens_per_query=round(total_prompt / total_queries, 1),
            avg_completion_tokens_per_query=round(total_completion / total_queries, 1),
            avg_tokens_per_query=round(total_tokens / total_queries, 1),
            total_cost_usd=round(total_cost, 6),
            avg_cost_per_query_usd=round(total_cost / total_queries, 6),
        )

    def clear(self) -> None:
        """Clear all recorded usage."""
        self._prompt_tokens.clear()
        self._completion_tokens.clear()