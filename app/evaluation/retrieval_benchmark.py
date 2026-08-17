"""Retrieval benchmark runner (Phase 8).

Compares retriever CONFIGURATIONS under identical conditions:
same corpus, same chunking (default strategy, real pipeline), same gold
evidence spans, same question set. Dense uses the hash model and the
in-memory vector repo; sparse uses pure-Python BM25 — infrastructure is
held out so differences are attributable to fusion mechanics (Phase 5
methodology, decisions D-030/D-031).

Configurations benchmarked:
  dense            — vector similarity only
  sparse_bm25      — lexical only
  hybrid_weighted  — min-max normalized weighted fusion
  hybrid_rrf       — Reciprocal Rank Fusion (production default)
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import RetrievalSettings
from app.evaluation.chunking_eval.corpus import CorpusDocument
from app.evaluation.chunking_eval.samples import ChunkingEvalSample
from app.evaluation.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.relevance import relevant_chunk_indices
from app.embeddings.base import EmbeddingModel
from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking.factory import build_chunker
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.cleaning import clean_parsed_document
from app.ingestion.domain import Chunk
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.structure import annotate_sections
from app.repositories.vector.base import VectorPayload, VectorPoint
from app.repositories.vector.in_memory_repository import InMemoryVectorRepository
from app.retrieval.base import Retriever
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import BM25Retriever, SparseChunkRecord

from datetime import datetime, timezone

DEFAULT_K_VALUES = (5, 10)
RETRIEVAL_DEPTH = 20


@dataclass(frozen=True)
class RetrieverBenchmarkResult:
    name: str
    config: dict[str, object]
    metrics: dict[str, float]
    per_sample_gold_sizes: list[int] = field(default_factory=list)


class _IndexedCorpus:
    """Corpus chunked once, indexed for both branches — shared across configs."""

    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        registry: ChunkingRegistry,
        embedding_model: EmbeddingModel,
    ) -> None:
        parser = MarkdownParser()
        chunker = build_chunker(registry.strategies[registry.default])
        self.texts: list[str] = []
        self.records: list[SparseChunkRecord] = []

        for document in corpus:
            parsed = parser.parse(document.text.encode(), source_uri=document.name)
            blocks = annotate_sections(clean_parsed_document(parsed)).blocks
            for chunk in chunker.chunk(blocks):
                self.texts.append(chunk.text)
                self.records.append(
                    SparseChunkRecord(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.uuid4(),
                        version_id=uuid.uuid4(),
                        content=chunk.text,
                        section=chunk.section,
                        source_uri=document.name,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                    )
                )

        # Dense index (in-memory vector repo + configured model)
        self.embedding_model = embedding_model
        self.vector_repo = InMemoryVectorRepository(collection="benchmark")
        self._build_dense_index()

        # Sparse index (pure BM25)
        self.bm25 = BM25Index(self.texts)

    def _build_dense_index(self) -> None:
        vectors = self.embedding_model.embed_documents(self.texts)
        now = datetime.now(timezone.utc)
        points = [
            VectorPoint(
                point_id=record.chunk_id,
                vector=vector,
                payload=VectorPayload(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    version_id=record.version_id,
                    owner_id=uuid.uuid4(),
                    is_active_version=True,
                    source_type="markdown",
                    content=record.content,
                    token_count=0,
                    created_at=now,
                    section=record.section,
                    source_uri=record.source_uri,
                ),
            )
            for record, vector in zip(self.records, vectors)
        ]
        import asyncio

        asyncio.get_event_loop().run_until_complete(self.vector_repo.upsert_points(points))

    def dense_retriever(self) -> DenseRetriever:
        service = EmbeddingService(self.embedding_model, cache=InMemoryCacheStore())
        return DenseRetriever(self.vector_repo, service)

    def sparse_retriever(self) -> BM25Retriever:
        return BM25Retriever(self.bm25, self.records)

    def gold_chunk_ids(self, sample: ChunkingEvalSample) -> set[uuid.UUID]:
        indices = relevant_chunk_indices(self.texts, sample.evidence)
        return {self.records[i].chunk_id for i in indices}


class RetrievalBenchmarkRunner:
    def __init__(
        self,
        corpus: Sequence[CorpusDocument],
        samples: Sequence[ChunkingEvalSample],
        chunking_registry: ChunkingRegistry,
        embedding_model: EmbeddingModel,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        tracker: object | None = None,
    ) -> None:
        self.samples = list(samples)
        self.k_values = k_values
        self.tracker = tracker
        self.indexed = _IndexedCorpus(corpus, chunking_registry, embedding_model)

    # ------------------------------------------------------------------ public
    def run_all(self) -> list[RetrieverBenchmarkResult]:
        return [
            self.run_configuration("dense", self.indexed.dense_retriever(), {}),
            self.run_configuration("sparse_bm25", self.indexed.sparse_retriever(), {}),
            self.run_configuration(
                "hybrid_weighted",
                self._hybrid(fusion_strategy="weighted"),
                {"fusion": "weighted", "dense_weight": 0.7, "sparse_weight": 0.3},
            ),
            self.run_configuration(
                "hybrid_rrf",
                self._hybrid(fusion_strategy="rrf"),
                {"fusion": "rrf", "rrf_k": 60},
            ),
        ]

    def run_configuration(
        self, name: str, retriever: Retriever, config: dict[str, object]
    ) -> RetrieverBenchmarkResult:
        import asyncio

        metric_series: dict[str, list[float]] = {}
        gold_sizes: list[int] = []
        for sample in self.samples:
            gold = self.indexed.gold_chunk_ids(sample)
            gold_sizes.append(len(gold))
            result = asyncio.get_event_loop().run_until_complete(
                retriever.retrieve(sample.question, top_k=RETRIEVAL_DEPTH)
            )
            ranked = [item.chunk_id for item in result.items]
            metrics = {"mrr": reciprocal_rank(ranked, gold)}
            for k in self.k_values:
                metrics[f"precision_at_{k}"] = precision_at_k(ranked, gold, k)
                metrics[f"recall_at_{k}"] = recall_at_k(ranked, gold, k)
                metrics[f"hit_rate_at_{k}"] = hit_at_k(ranked, gold, k)
            for key, value in metrics.items():
                metric_series.setdefault(key, []).append(value)

        aggregated = {key: round(mean(series), 4) for key, series in metric_series.items()}
        if self.tracker is not None:
            self.tracker.track_strategy(
                run_name=f"retrieval-{name}",
                params={"retriever": name, **{k: str(v) for k, v in config.items()}},
                metrics=aggregated,
            )
        return RetrieverBenchmarkResult(
            name=name, config=config, metrics=aggregated, per_sample_gold_sizes=gold_sizes
        )

    # ---------------------------------------------------------------- internals
    def _hybrid(self, fusion_strategy: str) -> HybridRetriever:
        settings = RetrievalSettings(
            fusion_strategy=fusion_strategy,  # type: ignore[arg-type]
            dense_top_k=RETRIEVAL_DEPTH,
            sparse_top_k=RETRIEVAL_DEPTH,
            final_top_k=RETRIEVAL_DEPTH,
            degrade_policy="strict",
        )
        return HybridRetriever(
            self.indexed.dense_retriever(), self.indexed.sparse_retriever(), settings
        )