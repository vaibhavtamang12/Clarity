"""Retrieval evaluation runner (Phase 21).

Runs the full retrieval evaluation pipeline:
1. Load evaluation dataset
2. Ingest corpus into real system (PostgreSQL + Qdrant)
3. Build four retriever configurations
4. For each question: retrieve, compute metrics, track
5. Aggregate results and generate comparison table

This uses REAL components (not hash embeddings), so it requires:
- PostgreSQL with ingested corpus
- Qdrant with real embeddings
- Real embedding model (bge-m3 or similar)
- Real reranker (bge-reranker-v2-m3)

The evaluation is expensive (model downloads, inference time) and is run
as a script, not in CI unit tests.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.evaluation.dataset_schema import EvaluationDataset, EvaluationQuestion
from app.evaluation.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.relevance import relevant_chunk_indices
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.service import EmbeddingService
from app.ingestion.chunking_registry import ChunkingRegistry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import FileStore
from app.models.document import Document
from app.models.enums import DocumentStatus, VersionStatus
from app.repositories.database import Database
from app.repositories.document import DocumentChunkRepository, DocumentRepository, DocumentVersionRepository
from app.repositories.vector.base import VectorFilter, VectorRepository
from app.retrieval.base import Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import PostgresSparseRetriever
from app.reranking.retriever import RerankedRetriever
from app.services.indexing_service import IndexingService

logger = get_logger(__name__)

DEFAULT_K_VALUES = (5, 10, 20)


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Result for a single question on a single retriever configuration."""

    question_id: str
    category: str
    retriever_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    num_retrieved: int = 0
    num_gold: int = 0


@dataclass(frozen=True)
class RetrievalEvalSummary:
    """Aggregated results for a retriever configuration."""

    retriever_name: str
    config: dict[str, str]
    metrics: dict[str, float]
    per_category: dict[str, dict[str, float]]
    num_questions: int
    total_latency_ms: float


class RetrievalEvaluationRunner:
    def __init__(
        self,
        database: Database,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        settings: Settings,
        chunking_registry: ChunkingRegistry,
        file_store: FileStore,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        tracker: object | None = None,
    ) -> None:
        self._database = database
        self._vector_repo = vector_repository
        self._embedding_service = embedding_service
        self._settings = settings
        self._chunking_registry = chunking_registry
        self._file_store = file_store
        self._k_values = k_values
        self._tracker = tracker

    async def run(
        self,
        dataset: EvaluationDataset,
        corpus_dir: Path,
    ) -> list[RetrievalEvalSummary]:
        """Run full retrieval evaluation."""
        # ---- 1. Ingest corpus -----------------------------------------------
        logger.info("ingesting_evaluation_corpus", corpus_dir=str(corpus_dir))
        document_map = await self._ingest_corpus(corpus_dir)
        logger.info("corpus_ingested", num_documents=len(document_map))

        # ---- 2. Build retriever configurations -------------------------------
        retrievers = await self._build_retrievers()
        logger.info("retrievers_built", configs=list(retrievers.keys()))

        # ---- 3. Run evaluation for each configuration ------------------------
        summaries: list[RetrievalEvalSummary] = []
        for retriever_name, retriever in retrievers.items():
            summary = await self._evaluate_retriever(
                retriever_name, retriever, dataset, document_map
            )
            summaries.append(summary)
            if self._tracker is not None:
                self._tracker.track_strategy(
                    run_name=f"retrieval-eval-{retriever_name}",
                    params=summary.config,
                    metrics=summary.metrics,
                )

        logger.info("retrieval_evaluation_complete", num_configs=len(summaries))
        return summaries

    # ------------------------------------------------------------------ internals
    async def _ingest_corpus(self, corpus_dir: Path) -> dict[str, uuid.UUID]:
        """Ingest all corpus documents into the system."""
        from app.ingestion.parsers.base import build_default_registry

        pipeline = IngestionPipeline(build_default_registry(), self._chunking_registry, self._settings)
        indexer = IndexingService(
            repository=self._vector_repo,
            embedding_pipeline=EmbeddingPipeline(self._embedding_service),
            dimension=self._embedding_service.dimension,
        )

        document_map: dict[str, uuid.UUID] = {}

        async with self._database.session() as session:
            # Create a synthetic owner for evaluation documents
            from app.repositories.user import UserRepository
            from app.utils.security import hash_password

            owner = await UserRepository(session).get_by_email("eval@example.com")
            if owner is None:
                owner = await UserRepository(session).create(
                    email="eval@example.com", hashed_password=hash_password("eval-password")
                )
                await session.commit()

            for md_file in corpus_dir.glob("*.md"):
                doc_name = md_file.name
                if doc_name in document_map:
                    continue  # already ingested

                content = md_file.read_bytes()
                document = await DocumentRepository(session).create(
                    owner_id=owner.id,
                    source_type="markdown",
                    title=doc_name,
                    source_uri=doc_name,
                )
                document.status = DocumentStatus.PENDING
                await session.flush()

                self._file_store.save(str(document.id), content)

                result = await pipeline.ingest(session=session, document=document, content=content)
                if result.version_id is not None and result.changed:
                    await indexer.index_version(session, result.version_id)

                document_map[doc_name] = document.id
                logger.info("document_ingested", document=doc_name, document_id=str(document.id))

            await session.commit()

        return document_map

    async def _build_retrievers(self) -> dict[str, Retriever]:
        """Build all four retriever configurations."""
        from app.reranking.factory import build_reranker
        from app.reranking.registry import load_reranker_registry

        dense = DenseRetriever(self._vector_repo, self._embedding_service)
        sparse = PostgresSparseRetriever(self._database)
        hybrid = HybridRetriever(dense, sparse, self._settings.retrieval)

        reranker_registry = load_reranker_registry()
        reranker = build_reranker(reranker_registry, self._settings.reranker)
        reranked = RerankedRetriever(
            hybrid,
            reranker,
            candidates=self._settings.reranker.candidates,
            top_n=self._settings.reranker.top_n,
        )

        return {
            "dense": dense,
            "sparse": sparse,
            "hybrid": hybrid,
            "hybrid_reranker": reranked,
        }

    async def _evaluate_retriever(
        self,
        retriever_name: str,
        retriever: Retriever,
        dataset: EvaluationDataset,
        document_map: dict[str, uuid.UUID],
    ) -> RetrievalEvalSummary:
        """Evaluate one retriever configuration on all questions."""
        results: list[RetrievalEvalResult] = []

        async with self._database.session() as session:
            chunks_repo = DocumentChunkRepository(session)

            for question in dataset.questions:
                result = await self._evaluate_question(
                    retriever_name, retriever, question, document_map, chunks_repo
                )
                results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_results(results)
        per_category = self._aggregate_by_category(results)
        total_latency = sum(r.latency_ms for r in results)

        summary = RetrievalEvalSummary(
            retriever_name=retriever_name,
            config={"retriever": retriever_name},
            metrics=aggregated,
            per_category=per_category,
            num_questions=len(results),
            total_latency_ms=total_latency,
        )
        logger.info(
            "retriever_evaluation_complete",
            retriever=retriever_name,
            mrr=aggregated.get("mrr", 0.0),
            recall_10=aggregated.get("recall_at_10", 0.0),
        )
        return summary

    async def _evaluate_question(
        self,
        retriever_name: str,
        retriever: Retriever,
        question: EvaluationQuestion,
        document_map: dict[str, uuid.UUID],
        chunks_repo: DocumentChunkRepository,
    ) -> RetrievalEvalResult:
        """Evaluate one question on one retriever."""
        # Resolve gold chunk IDs from evidence spans
        gold_chunk_ids: set[uuid.UUID] = set()
        for doc_name in question.expected_sources:
            doc_id = document_map.get(doc_name)
            if doc_id is None:
                continue
            # Get active version
            async with self._database.session() as session:
                versions_repo = DocumentVersionRepository(session)
                active = await versions_repo.get_active_for_document(doc_id)
                if active is None:
                    continue
                chunks = await chunks_repo.list_for_version(active.id, batch_size=500)

            # Match evidence spans to chunks
            chunk_texts = [c.content for c in chunks]
            for evidence in question.evidence:
                indices = relevant_chunk_indices(chunk_texts, [evidence.text])
                for idx in indices:
                    if idx < len(chunks):
                        gold_chunk_ids.add(chunks[idx].id)

        # Retrieve
        start = time.perf_counter()
        retrieval_result = await retriever.retrieve(question.question, top_k=max(self._k_values))
        latency_ms = (time.perf_counter() - start) * 1000

        # Compute metrics
        ranked_chunk_ids = [item.chunk_id for item in retrieval_result.items]
        metrics: dict[str, float] = {"mrr": reciprocal_rank(ranked_chunk_ids, gold_chunk_ids)}
        for k in self._k_values:
            metrics[f"precision_at_{k}"] = precision_at_k(ranked_chunk_ids, gold_chunk_ids, k)
            metrics[f"recall_at_{k}"] = recall_at_k(ranked_chunk_ids, gold_chunk_ids, k)
            metrics[f"hit_rate_at_{k}"] = hit_at_k(ranked_chunk_ids, gold_chunk_ids, k)

        return RetrievalEvalResult(
            question_id=question.id,
            category=question.category,
            retriever_name=retriever_name,
            metrics=metrics,
            latency_ms=latency_ms,
            num_retrieved=len(retrieval_result.items),
            num_gold=len(gold_chunk_ids),
        )

    def _aggregate_results(self, results: list[RetrievalEvalResult]) -> dict[str, float]:
        """Aggregate metrics across all questions."""
        if not results:
            return {}
        metric_keys = results[0].metrics.keys()
        aggregated: dict[str, float] = {}
        for key in metric_keys:
            values = [r.metrics[key] for r in results if key in r.metrics]
            if values:
                aggregated[key] = round(mean(values), 4)
        return aggregated

    def _aggregate_by_category(
        self, results: list[RetrievalEvalResult]
    ) -> dict[str, dict[str, float]]:
        """Aggregate metrics per question category."""
        by_category: dict[str, list[RetrievalEvalResult]] = {}
        for result in results:
            by_category.setdefault(result.category, []).append(result)

        per_category: dict[str, dict[str, float]] = {}
        for category, category_results in by_category.items():
            per_category[category] = self._aggregate_results(category_results)
        return per_category