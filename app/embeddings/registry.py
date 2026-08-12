"""Embedding model registry (ADR-002).

Models are configuration, not code: each entry declares everything the
embedding pipeline needs (dimension, token budget, normalization, query
prefix). Changing the default model is an env-var change, not a code change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.exceptions import ConfigurationError
from app.utils.yaml import read_yaml

DEFAULT_REGISTRY_PATH = Path("configs/embeddings.yaml")


class EmbeddingModelConfig(BaseModel):
    model_id: str
    dimension: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    normalize: bool = True
    query_prefix: str = ""


class EmbeddingRegistry(BaseModel):
    default: str
    models: dict[str, EmbeddingModelConfig]


def load_embedding_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> EmbeddingRegistry:
    raw = read_yaml(path)
    try:
        registry = EmbeddingRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid embedding registry at {path}: {exc}") from exc
    if registry.default not in registry.models:
        raise ConfigurationError(
            f"Embedding registry default '{registry.default}' is not defined in models"
        )
    return registry