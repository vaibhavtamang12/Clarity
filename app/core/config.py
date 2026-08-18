"""Application configuration — single source of truth for all runtime settings.

Values come from environment variables (optionally a `.env` file) using a
double-underscore nested delimiter, e.g. ``DATABASE__HOST``.

Design notes:
- Every external dependency is configured here; nothing is hard-coded (Rule 3).
- Sub-settings are plain Pydantic models; only the root is a BaseSettings.
- ``get_settings()`` is cached; tests clear the cache to override values.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"


class AppSettings(BaseModel):
    name: str = "rag-knowledge-platform"
    version: str = "0.2.0"
    environment: Environment = Environment.DEV
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        level = v.upper()
        if level not in ALLOWED_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(ALLOWED_LOG_LEVELS)}")
        return level


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "rag_user"
    password: str = "rag_password"
    name: str = "rag_platform"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @field_validator("port", "pool_size", "max_overflow")
    @classmethod
    def _positive(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("must be within 1..65535")
        return v

    @property
    def async_url(self) -> str:
        """URL for the async SQLAlchemy engine (asyncpg driver)."""
        creds = f"{quote_plus(self.user)}:{quote_plus(self.password)}"
        return f"postgresql+asyncpg://{creds}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    timeout_seconds: float = 2.0

    @property
    def url(self) -> str:
        creds = f":{quote_plus(self.password)}@" if self.password else ""
        return f"redis://{creds}{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    collection_prefix: str = "rag"
    api_key: SecretStr | None = None
    timeout_seconds: float = 10.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class MLflowSettings(BaseModel):
    tracking_uri: str = "http://localhost:5001"
    experiment_name: str = "rag-platform"
    enabled: bool = True


class EmbeddingSettings(BaseModel):
    default_model: str = "bge_m3"  # key into configs/embeddings.yaml
    registry_path: str = "configs/embeddings.yaml"
    batch_size: int = 32
    device: str = "cpu"
    max_retries: int = 2                  # transient inference failures
    retry_base_delay_seconds: float = 0.5  


class RerankerSettings(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = True
    candidates: int = 50
    top_n: int = 10
    batch_size: int = 16
    device: str = "cpu"


class LLMSettings(BaseModel):
    provider: Literal["openai_compatible", "anthropic", "ollama"] = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: SecretStr | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries: int = 2


class RetrievalSettings(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 10
    fusion_strategy: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    degrade_policy: Literal["strict", "degrade"] = "degrade"
    branch_timeout_seconds: float = 2.0

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def _positive_weight(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weights must be > 0")
        return v

    @field_validator("final_top_k")
    @classmethod
    def _sane_top_k(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("final_top_k must be within 1..100")
        return v


class IngestionSettings(BaseModel):
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".md", ".txt", ".html"]
    )
    allowed_url_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    max_attempts: int = 3
    retry_base_delay_seconds: float = 2.0
    parser_timeout_seconds: float = 120.0

    @field_validator("max_attempts")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be >= 1")
        return v

    @field_validator("allowed_extensions")
    @classmethod
    def _normalized_extensions(cls, v: list[str]) -> list[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in v]


class CacheSettings(BaseModel):
    enabled: bool = True
    rag_response_ttl_seconds: int = 3600
    embedding_ttl_seconds: int = 86400


class SecuritySettings(BaseModel):
    rate_limit_per_minute: int = 60


class Settings(BaseSettings):
    """Root settings. Env vars use the nested delimiter: ``DATABASE__HOST``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)

    @property
    def is_production(self) -> bool:
        return self.app.environment == Environment.PROD

class QueryTransformSettings(BaseModel):
    enabled: bool = True
    history_window: int = 3                # conversation turns given to the rewriter
    rewrite_max_output_tokens: int = 64
    expansion_min_query_tokens: int = 2    # queries with ≤ this many tokens may expand
    expansion_max_terms: int = 8
    decompose_min_tokens: int = 12         # only long queries may decompose
    temperature: float = 0.1
    max_output_chars: int = 300

    @field_validator("history_window", "expansion_max_terms")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

class GenerationSettings(BaseModel):
    max_context_tokens: int = 3000
    max_answer_tokens: int = 512
    temperature: float = 0.2
    max_citations: int = 8
    repair_attempts: int = 1           # schema-repair retries (ADR-005)
    fallback_notice: str = (
        "The answer generator was unavailable. The most relevant retrieved "
        "passages are provided as citations instead."
    )

class WorkerSettings(BaseModel):
    group_name: str = "ingestion-workers"
    stream_key: str = "queue:ingest"
    dlq_key: str = "queue:ingest:dlq"
    block_ms: int = 1000                 # stream read block
    poll_interval_seconds: float = 2.0   # DB sweep cadence (fallback + safety net)
    lock_ttl_seconds: int = 600          # per-document lock expiry
    shutdown_grace_seconds: float = 10.0

    @field_validator("block_ms")
    @classmethod
    def _positive_block(cls, v: int) -> int:
        if v < 0:
            raise ValueError("block_ms must be >= 0")
        return v

class GroundingSettings(BaseModel):
    poor_grounding_threshold: float = 0.6   # score below this triggers policy action
    max_regeneration_attempts: int = 1        # how many times to regenerate with stricter prompt
    retrieve_more_context_enabled: bool = True
    context_expansion_factor: float = 1.5     # 3000 → 4500 tokens on retrieve_more
    judge_unsupported: bool = True            # send unsupported-with-markers claims to judge
    max_claim_chars: int = 500
    max_evidence_chars: int = 2000
    uncertainty_notice: str = (
        "The answer below may contain claims not fully supported by the retrieved "
        "evidence. Please verify against the cited sources before acting on it."
    )

    @field_validator("poor_grounding_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return v


# In Settings root, add:
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)


# In Settings root, add:
    generation: GenerationSettings = Field(default_factory=GenerationSettings)


# In Settings root, add:
    query_transform: QueryTransformSettings = Field(default_factory=QueryTransformSettings)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()