"""Composition root (decision D-083).

Lives OUTSIDE the layered packages so it may import everything without
breaking the import-linter contracts. One function wires settings → services;
tests inject overrides (in-memory vector repo, mock LLM) through the same
seam production uses for configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api.auth import ApiKeyAuthProvider, AuthProvider
from app.api.rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.conversation.history import HistorySelector
from app.core.config import Settings
from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.factory import build_embedding_model
from app.embeddings.naming import collection_name
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.generation.context import ContextBuilder
from app.generation.generator import RAGGenerator
from app.generation.pipeline import RAGPipeline
from app.ingestion.chunking_registry import load_chunking_registry
from app.ingestion.parsers.base import build_default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import FileStore, LocalFileStore
from app.llm.base import LLMProvider
from app.llm.factory import build_llm_provider
from app.reranking.factory import build_reranker
from app.reranking.registry import load_reranker_registry
from app.reranking.retriever import RerankedRetriever
from app.repositories.cache import RedisCacheStore
from app.repositories.database import Database
from app.repositories.job_queue import InMemoryJobQueue, RedisStreamJobQueue
from app.repositories.locks import InMemoryDocumentLock, RedisDocumentLock
from app.repositories.vector.base import VectorRepository
from app.repositories.vector.qdrant_client import build_qdrant_client
from app.repositories.vector.qdrant_repository import QdrantVectorRepository
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query.service import build_query_transform_service
from app.retrieval.sparse import PostgresSparseRetriever
from app.services.chat_service import ChatService
from app.services.index_state import (
    IndexStateNotifier,
    LocalIndexStateNotifier,
    RedisIndexStateNotifier,
)
from app.services.indexing_service import IndexingService
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from app.services.job_state import InMemoryJobStateStore, JobStateStore, RedisJobStateStore
from app.services.response_cache import ResponseCache
from app.services.versioning_service import VersioningService
from app.workers.protocols import DeadLetterSink, DocumentLockManager, JobQueue


@dataclass
class Platform:
    settings: Settings
    database: Database
    file_store: FileStore
    embedding_service: EmbeddingService
    vector_repository: VectorRepository
    ingestion_service: IngestionService
    job_runner: IngestionJobRunner
    job_queue: JobQueue                       # Phase 19
    document_locks: DocumentLockManager       # Phase 19
    dlq: DeadLetterSink                       # Phase 19
    versioning_service: VersioningService
    retriever: object
    rag_pipeline: RAGPipeline
    chat_service: ChatService
    auth_provider: AuthProvider
    rate_limiter: RateLimiter
    job_state_store: JobStateStore
    response_cache: ResponseCache


def build_platform(
    settings: Settings,
    database: Database,
    *,
    qdrant_client: object | None = None,
    redis_client: object | None = None,
    vector_repository: VectorRepository | None = None,
    file_store: FileStore | None = None,
    llm_provider: LLMProvider | None = None,
) -> Platform:
    # ---- registries -----------------------------------------------------------
    chunking_registry = load_chunking_registry()
    embedding_registry = load_embedding_registry(settings.embedding.registry_path)
    reranker_registry = load_reranker_registry()

    # ---- Redis-backed infrastructure with graceful fallbacks (D-101) ----------
    if redis_client is not None:
        cache_store = RedisCacheStore(redis_client)  # type: ignore[arg-type]
        index_state: IndexStateNotifier = RedisIndexStateNotifier(redis_client)  # type: ignore[arg-type]
        rate_limiter: RateLimiter = RedisRateLimiter(redis_client)  # type: ignore[arg-type]
        job_state_store: JobStateStore = RedisJobStateStore(redis_client)  # type: ignore[arg-type]
        job_queue: JobQueue = RedisStreamJobQueue(
            redis_client,  # type: ignore[arg-type]
            stream_key=settings.worker.stream_key,
            group_name=settings.worker.group_name,
            dlq_key=settings.worker.dlq_key,
        )
        document_locks: DocumentLockManager = RedisDocumentLock(
            redis_client,  # type: ignore[arg-type]
            ttl_seconds=settings.worker.lock_ttl_seconds,
        )
        dlq: DeadLetterSink = job_queue  # type: ignore[assignment]
    else:
        cache_store = InMemoryCacheStore()
        index_state = LocalIndexStateNotifier()
        rate_limiter = InMemoryRateLimiter()
        job_state_store = InMemoryJobStateStore()
        job_queue = InMemoryJobQueue()
        document_locks = InMemoryDocumentLock()
        dlq = job_queue  # type: ignore[assignment]

    # ---- ML components ---------------------------------------------------------
    embedding_model = build_embedding_model(embedding_registry, settings.embedding)
    embedding_service = EmbeddingService(
        embedding_model,
        cache=cache_store,
        cache_ttl_seconds=settings.cache.embedding_ttl_seconds,
        max_retries=settings.embedding.max_retries,
        retry_base_delay_seconds=settings.embedding.retry_base_delay_seconds,
    )
    if vector_repository is None:
        client = qdrant_client or build_qdrant_client(settings.qdrant)
        vector_repository = QdrantVectorRepository(
            client,
            collection_name(settings.qdrant.collection_prefix, settings.embedding.default_model),
        )
    llm = llm_provider or build_llm_provider(settings.llm)
    reranker = build_reranker(reranker_registry, settings.reranker)

    # ---- ingestion ---------------------------------------------------------------
    store = file_store or LocalFileStore(Path("data/uploads"))
    ingestion_pipeline = IngestionPipeline(build_default_registry(), chunking_registry, settings)
    ingestion_service = IngestionService(store, chunking_registry, settings, job_queue=job_queue)
    indexer = IndexingService(
        repository=vector_repository,
        embedding_pipeline=EmbeddingPipeline(embedding_service),
        dimension=embedding_service.dimension,
        index_state=index_state,
    )
    job_runner = IngestionJobRunner(
        database,
        ingestion_pipeline,
        store,
        settings,
        indexer=indexer,
        job_state_store=job_state_store,
        locks=document_locks,
        dlq=dlq,
    )
    versioning_service = VersioningService(
        file_store=store,
        chunking_registry=chunking_registry,
        settings=settings,
        indexer=indexer,
        index_state=index_state,
    )

    # ---- retrieval stack ------------------------------------------------------------
    dense = DenseRetriever(vector_repository, embedding_service)
    sparse = PostgresSparseRetriever(database)
    hybrid = HybridRetriever(dense, sparse, settings.retrieval)
    retriever: object = hybrid
    if settings.reranker.enabled:
        retriever = RerankedRetriever(
            hybrid,
            reranker,
            candidates=settings.reranker.candidates,
            top_n=settings.reranker.top_n,
        )

    # ---- generation -------------------------------------------------------------------
    generator = RAGGenerator(llm, settings.generation)
    rag_pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        generator=generator,
        context_builder=ContextBuilder(),
        settings=settings.generation,
        transform_service=build_query_transform_service(llm, settings.query_transform),
        grounding_settings=settings.grounding,
    )

    # ---- conversation + auth -------------------------------------------------------------
    response_cache = ResponseCache(
        store=cache_store,
        index_state=index_state,
        settings=settings.cache,
        collection_scope=vector_repository.collection,
    )
    chat_service = ChatService(
        rag_pipeline=rag_pipeline,
        history_selector=HistorySelector(embedding_service),
        response_cache=response_cache,
    )
    auth_provider = ApiKeyAuthProvider()

    return Platform(
        settings=settings,
        database=database,
        file_store=store,
        embedding_service=embedding_service,
        vector_repository=vector_repository,
        ingestion_service=ingestion_service,
        job_runner=job_runner,
        job_queue=job_queue,
        document_locks=document_locks,
        dlq=dlq,
        versioning_service=versioning_service,
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        chat_service=chat_service,
        auth_provider=auth_provider,
        rate_limiter=rate_limiter,
        job_state_store=job_state_store,
        response_cache=response_cache,
    )