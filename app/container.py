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
from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.embeddings.caching import InMemoryCacheStore
from app.embeddings.factory import build_embedding_model
from app.embeddings.pipeline import EmbeddingPipeline
from app.embeddings.registry import load_embedding_registry
from app.embeddings.service import EmbeddingService
from app.embeddings.naming import collection_name
from app.conversation.history import HistorySelector
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
from app.repositories.database import Database
from app.repositories.vector.base import VectorRepository
from app.repositories.vector.qdrant_client import build_qdrant_client
from app.repositories.vector.qdrant_repository import QdrantVectorRepository
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query.service import build_query_transform_service
from app.retrieval.sparse import PostgresSparseRetriever
from app.services.chat_service import ChatService
from app.services.index_state import LocalIndexStateNotifier
from app.services.indexing_service import IndexingService
from app.services.ingestion_job_runner import IngestionJobRunner
from app.services.ingestion_service import IngestionService
from app.services.versioning_service import VersioningService


@dataclass
class Platform:
    settings: Settings
    database: Database
    file_store: FileStore
    embedding_service: EmbeddingService
    vector_repository: VectorRepository
    ingestion_service: IngestionService
    job_runner: IngestionJobRunner
    versioning_service: VersioningService
    retriever: object          # Retriever protocol
    rag_pipeline: RAGPipeline
    chat_service: ChatService
    auth_provider: AuthProvider


def build_platform(
    settings: Settings,
    database: Database,
    *,
    qdrant_client: object | None = None,
    vector_repository: VectorRepository | None = None,
    file_store: FileStore | None = None,
    llm_provider: LLMProvider | None = None,
) -> Platform:
    # ---- registries -----------------------------------------------------------
    chunking_registry = load_chunking_registry()
    embedding_registry = load_embedding_registry(settings.embedding.registry_path)
    reranker_registry = load_reranker_registry()

    # ---- ML components ---------------------------------------------------------
    embedding_model = build_embedding_model(embedding_registry, settings.embedding)
    embedding_service = EmbeddingService(
        embedding_model,
        cache=InMemoryCacheStore(),  # Redis cache store lands in Phase 18
        cache_ttl_seconds=settings.cache.embedding_ttl_seconds,
        max_retries=settings.embedding.max_retries,
        retry_base_delay_seconds=settings.embedding.retry_base_delay_seconds,
    )
    if vector_repository is None:
        client = qdrant_client or build_qdrant_client(settings.qdrant)
        vector_repository = QdrantVectorRepository(
            client, collection_name(settings.qdrant.collection_prefix, settings.embedding.default_model)
        )
    llm = llm_provider or build_llm_provider(settings.llm)
    reranker = build_reranker(reranker_registry, settings.reranker)

    # ---- ingestion ---------------------------------------------------------------
    store = file_store or LocalFileStore(Path("data/uploads"))
    ingestion_pipeline = IngestionPipeline(build_default_registry(), chunking_registry, settings)
    ingestion_service = IngestionService(store, chunking_registry, settings)
    indexer = IndexingService(
        repository=vector_repository,
        embedding_pipeline=EmbeddingPipeline(embedding_service),
        dimension=embedding_service.dimension,
    )
    index_state = LocalIndexStateNotifier()
    job_runner = IngestionJobRunner(database, ingestion_pipeline, store, settings, indexer=indexer)
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
    chat_service = ChatService(
        rag_pipeline=rag_pipeline,
        history_selector=HistorySelector(embedding_service),
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
        versioning_service=versioning_service,
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        chat_service=chat_service,
        auth_provider=auth_provider,
    )