"""Reranking benchmark runner (Phase 9).

Compares WITHOUT vs WITH reranking on the shared Phase 5/8 harness:
same corpus, same chunking, same gold evidence spans, same questions.

Measured per configuration:
- Retrieval quality: MRR, Recall@K, Precision@K, Hit Rate
- Latency: total wall time per query (mean + p95) and rerank stage time
- Cost proxy: total (query, chunk) pairs scored by the reranker

The reranker under test here is the deterministic hash double — this
validates pipeline mechanics and latency accounting (D-052). The real
cross-encoder's quality delta is a Phase 21 measurement.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import RetrievalSettings
from app.evaluation.chunking_eval.corpus import CorpusDocument
from app.evaluation.chunking_eval.samples import ChunkingEvalSample
from app.evaluation.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.relevance import relevant_chunk_indices
from app.evaluation.retrieval_benchmark import RETRIEVAL_DEPTH, _IndexedCorpus
from app.embeddings.base import EmbeddingModel
from app.ingestion.chunking_registry import ChunkingRegistry
from app.reranking.base import Reranker
from app.reranking.retriever import RerankedRetriever
from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever

DEFAULT_K_VALUES = (5, 10)
RERANK_CANDIDATES = 20
RERANK_TOP_N = 10


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass(frozen=True)
class RerankBenchmarkResult:
    name: str
    metrics: dict[str, float]
    latency_stats: dict[str, float]
    pairs_scored: int


class RerankingBenchmarkRunner:
    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        samples: Sequence[ChunkingEvalSample],
        chunking_registry: ChunkingRegistry,
        embedding_model: EmbeddingModel,
        reranker: Reranker,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        tracker: object | None = None,
    ) -> None:
        self.samples = list(samples)
        self.k_values = k_values
        self.reranker = reranker
        self.tracker = tracker
        self.indexed = _IndexedCorpus(corpus, chunking_registry, embedding_model)

    # ------------------------------------------------------------------ configs
    def _base_hybrid(self) -> HybridRetriever:
        settings = RetrievalSettings(
            fusion_strategy="rrf",
            dense_top_k=RETRIEVAL_DEPTH,
            sparse_top_k=RETRIEVAL_DEPTH,
            final_top_k=RETRIEVAL_DEPTH,
            degrade_policy="strict",
        )
        return HybridRetriever(
            self.indexed.dense_retriever(), self.indexed.sparse_retriever(), settings
        )

    def configurations(self) -> list[tuple[str, Retriever]]:
        return [
            ("hybrid_rrf", self._base_hybrid()),
            (
                "hybrid_rrf_reranked",
                RerankedRetriever(
                    self._base_hybrid(),
                    self.reranker,
                    candidates=RERANK_CANDIDATES,
                    top_n=RERANK_TOP_N,
                ),
            ),
        ]

    # -------------------------------------------------------------------- runs
    async def run_all_async(self) -> list[RerankBenchmarkResult]:
        await self.indexed.build()
        return [
            await self.run_configuration(name, retriever)
            for name, retriever in self.configurations()
        ]

    async def run_configuration(self, name: str, retriever: Retriever) -> RerankBenchmarkResult:
        metric_series: dict[str, list[float]] = {}
        total_latencies_ms: list[float] = []
        rerank_latencies_ms: list[float] = []
        pairs_scored = 0

        for sample in self.samples:
            gold_indices = relevant_chunk_indices(self.indexed.texts, sample.evidence)
            gold = {self.indexed.records[i].chunk_id for i in gold_indices}

            start = time.perf_counter()
            result = await retriever.retrieve(sample.question, top_k=RETRIEVAL_DEPTH)
            total_latencies_ms.append((time.perf_counter() - start) * 1000)

            if "rerank" in result.metadata.branch_latencies_ms:
                rerank_latencies_ms.append(result.metadata.branch_latencies_ms["rerank"])
            pairs_scored += result.metadata.counts.get("rerank_candidates", 0)

            ranked = [item.chunk_id for item in result.items]
            metrics = {"mrr": reciprocal_rank(ranked, gold)}
            for k in self.k_values:
                metrics[f"precision_at_{k}"] = precision_at_k(ranked, gold, k)
                metrics[f"recall_at_{k}"] = recall_at_k(ranked, gold, k)
                metrics[f"hit_rate_at_{k}"] = hit_at_k(ranked, gold, k)
            for key, value in metrics.items():
                metric_series.setdefault(key, []).append(value)

        aggregated = {key: round(mean(series), 4) for key, series in metric_series.items()}
        latency_stats = {
            "mean_total_ms": round(mean(total_latencies_ms), 2),
            "p95_total_ms": round(percentile(total_latencies_ms, 0.95), 2),
            "mean_rerank_ms": round(mean(rerank_latencies_ms), 2),
            "p95_rerank_ms": round(percentile(rerank_latencies_ms, 0.95), 2),
        }
        if self.tracker is not None:
            self.tracker.track_strategy(
                run_name=f"reranking-{name}",
                params={"configuration": name, "reranker": self.reranker.model_key},
                metrics={**aggregated, **latency_stats, "pairs_scored": float(pairs_scored)},
            )
        return RerankBenchmarkResult(
            name=name, metrics=aggregated, latency_stats=latency_stats, pairs_scored=pairs_scored
        )