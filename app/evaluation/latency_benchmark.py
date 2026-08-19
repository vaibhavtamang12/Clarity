"""Latency and cost benchmark runner (Phase 23).

Runs queries through the full RAG pipeline multiple times to collect
latency distributions and token usage. Produces:
- Per-stage latency statistics (p50/p95/p99/mean/min/max)
- Token usage statistics (prompt/completion/total per query)
- Estimated cost per query based on model pricing
- Comparison across configurations (if multiple are provided)

This is EXPENSIVE (real LLM calls) — run as a script, not in CI tests.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.evaluation.cost_tracker import CostTracker, TokenUsageStats
from app.evaluation.dataset_schema import EvaluationDataset, EvaluationQuestion
from app.evaluation.latency_tracker import LatencyStats, LatencyTracker
from app.generation.pipeline import RAGPipeline
from app.repositories.database import Database

logger = get_logger(__name__)


@dataclass(frozen=True)
class LatencyBenchmarkResult:
    """Results for one configuration."""

    configuration: str
    latency_stats: list[LatencyStats]
    total_latency_stats: LatencyStats
    token_stats: TokenUsageStats
    num_queries: int
    num_iterations: int


class LatencyBenchmarkRunner:
    def __init__(
        self,
        database: Database,
        rag_pipelines: dict[str, RAGPipeline],
        settings: Settings,
        tracker: object | None = None,
    ) -> None:
        self._database = database
        self._pipelines = rag_pipelines
        self._settings = settings
        self._tracker = tracker

    async def run(
        self,
        dataset: EvaluationDataset,
        iterations: int = 3,
        warmup_iterations: int = 1,
    ) -> list[LatencyBenchmarkResult]:
        """Run latency benchmark across all configurations.

        Args:
            dataset: Evaluation dataset with questions
            iterations: Number of measurement iterations per question
            warmup_iterations: Number of warmup iterations (not measured)
        """
        results: list[LatencyBenchmarkResult] = []

        for config_name, pipeline in self._pipelines.items():
            logger.info(
                "starting_latency_benchmark",
                configuration=config_name,
                iterations=iterations,
            )
            result = await self._benchmark_configuration(
                config_name, pipeline, dataset, iterations, warmup_iterations
            )
            results.append(result)

            if self._tracker is not None:
                self._tracker.track_strategy(
                    run_name=f"latency-bench-{config_name}",
                    params={
                        "configuration": config_name,
                        "iterations": iterations,
                    },
                    metrics={
                        "p95_total_ms": result.total_latency_stats.p95_ms,
                        "avg_cost_per_query": result.token_stats.avg_cost_per_query_usd,
                    },
                )

        logger.info("latency_benchmark_complete", num_configs=len(results))
        return results

    async def _benchmark_configuration(
        self,
        config_name: str,
        pipeline: RAGPipeline,
        dataset: EvaluationDataset,
        iterations: int,
        warmup_iterations: int,
    ) -> LatencyBenchmarkResult:
        """Benchmark one configuration."""
        latency_tracker = LatencyTracker()
        cost_tracker = CostTracker()

        # Set model for cost tracking
        model_name = self._settings.llm.model
        cost_tracker.set_model(model_name)

        for question in dataset.questions:
            # Warmup iterations (not measured)
            for _ in range(warmup_iterations):
                await pipeline.answer(question.question)

            # Measurement iterations
            for _ in range(iterations):
                start = time.perf_counter()
                response = await pipeline.answer(question.question)
                total_latency_ms = (time.perf_counter() - start) * 1000

                # Record stage latencies
                latency_tracker.record_from_dict(response.stage_latencies_ms)
                latency_tracker.record("total", total_latency_ms)

                # Record token usage
                if response.token_usage:
                    cost_tracker.record(
                        response.token_usage.prompt_tokens,
                        response.token_usage.completion_tokens,
                    )

        latency_stats = latency_tracker.get_all_stats()
        total_stats = latency_tracker.get_total_stats()
        token_stats = cost_tracker.get_stats()

        logger.info(
            "configuration_benchmark_complete",
            configuration=config_name,
            p95_total_ms=total_stats.p95_ms,
            avg_tokens_per_query=token_stats.avg_tokens_per_query,
        )

        return LatencyBenchmarkResult(
            configuration=config_name,
            latency_stats=latency_stats,
            total_latency_stats=total_stats,
            token_stats=token_stats,
            num_questions=len(dataset.questions),
            num_iterations=iterations,
        )