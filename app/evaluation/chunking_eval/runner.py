"""Chunking experiment runner (Phase 5).

For every strategy: run the REAL ingestion path (parse → clean → annotate →
chunk) over the corpus, build a BM25 index over the resulting chunks, answer
every gold question, and score the rankings. The retriever is held constant
(lexical BM25) so differences in metrics are attributable to chunking alone.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.evaluation.chunking_eval.corpus import CorpusDocument
from app.evaluation.chunking_eval.samples import ChunkingEvalSample
from app.evaluation.metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.relevance import relevant_chunk_indices
from app.ingestion.chunking.factory import build_chunker
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.cleaning import clean_parsed_document
from app.ingestion.domain import Chunk
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.structure import annotate_sections
from app.ingestion.tokens import HeuristicTokenCounter, TokenCounter
from app.retrieval.bm25 import BM25Index

DEFAULT_K_VALUES = (5, 10)
DEFAULT_RETRIEVAL_DEPTH = 20


@dataclass(frozen=True)
class ChunkSetStats:
    n_chunks: int
    mean_tokens: float
    median_tokens: float
    max_tokens: int
    min_tokens: int
    pct_below_min_tokens: float


@dataclass(frozen=True)
class SampleOutcome:
    sample_id: str
    question: str
    n_relevant_chunks: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    config: dict[str, object]
    metrics: dict[str, float]
    stats: ChunkSetStats
    samples: list[SampleOutcome] = field(default_factory=list)


def compute_chunk_stats(token_counts: Sequence[int], min_tokens: int) -> ChunkSetStats:
    counts = list(token_counts)
    if not counts:
        return ChunkSetStats(0, 0.0, 0.0, 0, 0, 0.0)
    below = sum(1 for c in counts if c < min_tokens)
    return ChunkSetStats(
        n_chunks=len(counts),
        mean_tokens=round(statistics.mean(counts), 1),
        median_tokens=float(statistics.median(counts)),
        max_tokens=max(counts),
        min_tokens=min(counts),
        pct_below_min_tokens=round(100 * below / len(counts), 1),
    )


class ChunkingExperimentRunner:
    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        samples: Sequence[ChunkingEvalSample],
        chunking_registry: ChunkingRegistry,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH,
        token_counter: TokenCounter | None = None,
        tracker: object | None = None,  # ExperimentTracker protocol; kept loose to avoid cycles
    ) -> None:
        self.corpus = list(corpus)
        self.samples = list(samples)
        self.registry = chunking_registry
        self.k_values = k_values
        self.retrieval_depth = retrieval_depth
        self.counter = token_counter or HeuristicTokenCounter()
        self.tracker = tracker
        self._parser = MarkdownParser()

    # ------------------------------------------------------------------ public
    def run_all(self) -> list[StrategyResult]:
        return [self.run_strategy(name) for name in self.registry.strategies]

    def run_strategy(self, strategy_name: str) -> StrategyResult:
        config = self.registry.strategies[strategy_name]
        chunker = build_chunker(config, self.counter)

        chunk_texts: list[str] = []
        chunk_origins: list[tuple[str, Chunk]] = []
        for document in self.corpus:
            parsed = self._parser.parse(document.text.encode(), source_uri=document.name)
            blocks = annotate_sections(clean_parsed_document(parsed)).blocks
            for chunk in chunker.chunk(blocks):
                chunk_texts.append(chunk.text)
                chunk_origins.append((document.name, chunk))

        index = BM25Index(chunk_texts)
        sample_outcomes: list[SampleOutcome] = []
        metric_series: dict[str, list[float]] = {}

        for sample in self.samples:
            relevant = relevant_chunk_indices(chunk_texts, sample.evidence)
            ranked = [sd.index for sd in index.search(sample.question, self.retrieval_depth)]
            metrics: dict[str, float] = {"mrr": reciprocal_rank(ranked, relevant)}
            for k in self.k_values:
                metrics[f"precision_at_{k}"] = precision_at_k(ranked, relevant, k)
                metrics[f"recall_at_{k}"] = recall_at_k(ranked, relevant, k)
                metrics[f"hit_rate_at_{k}"] = hit_at_k(ranked, relevant, k)
            for name, value in metrics.items():
                metric_series.setdefault(name, []).append(value)
            sample_outcomes.append(
                SampleOutcome(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    n_relevant_chunks=len(relevant),
                    metrics=metrics,
                )
            )

        aggregated = {name: round(mean(series), 4) for name, series in metric_series.items()}
        stats = compute_chunk_stats(
            [self.counter.count(text) for text in chunk_texts], config.min_tokens
        )
        result = StrategyResult(
            strategy=strategy_name,
            config=config.model_dump(mode="json"),
            metrics=aggregated,
            stats=stats,
            samples=sample_outcomes,
        )
        if self.tracker is not None:
            self.tracker.track_strategy(
                run_name=f"chunking-{strategy_name}",
                params={"strategy": strategy_name, **{k: str(v) for k, v in result.config.items()}},
                metrics={**aggregated, "n_chunks": float(stats.n_chunks)},
            )
        return result