"""Latency tracking and percentile computation (Phase 23).

Collects per-stage latency samples across multiple query executions and
computes p50/p95/p99/mean/min/max statistics. This feeds the latency table
in the benchmark report and identifies bottlenecks for optimization (Phase 29).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatencyStats:
    """Latency statistics for a single stage."""

    stage: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def percentile(values: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


class LatencyTracker:
    """Collects latency samples per stage and computes statistics."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, latency_ms: float) -> None:
        """Record a latency sample for a stage."""
        self._samples[stage].append(latency_ms)

    def record_from_dict(self, stage_latencies: dict[str, float]) -> None:
        """Record multiple stages from a dict (e.g., RAGResponse.stage_latencies_ms)."""
        for stage, latency_ms in stage_latencies.items():
            self.record(stage, latency_ms)

    def get_stats(self, stage: str) -> LatencyStats:
        """Compute statistics for a single stage."""
        samples = self._samples.get(stage, [])
        if not samples:
            return LatencyStats(
                stage=stage, count=0, mean_ms=0.0, min_ms=0.0, max_ms=0.0,
                p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
            )
        return LatencyStats(
            stage=stage,
            count=len(samples),
            mean_ms=round(statistics.mean(samples), 2),
            min_ms=round(min(samples), 2),
            max_ms=round(max(samples), 2),
            p50_ms=round(percentile(samples, 50), 2),
            p95_ms=round(percentile(samples, 95), 2),
            p99_ms=round(percentile(samples, 99), 2),
        )

    def get_all_stats(self) -> list[LatencyStats]:
        """Compute statistics for all stages."""
        return [self.get_stats(stage) for stage in sorted(self._samples.keys())]

    def get_total_stats(self) -> LatencyStats:
        """Compute statistics for total latency (sum of all stages per sample)."""
        if not self._samples:
            return LatencyStats(
                stage="total", count=0, mean_ms=0.0, min_ms=0.0, max_ms=0.0,
                p50_ms=0.0, p95_ms=0.0, p99_ms=0.0,
            )
        # Total latency is recorded as a separate "total" stage
        return self.get_stats("total")

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()

    @property
    def stages(self) -> list[str]:
        """List of stages with samples."""
        return sorted(self._samples.keys())

    @property
    def total_samples(self) -> int:
        """Total number of samples across all stages."""
        return sum(len(samples) for samples in self._samples.values())