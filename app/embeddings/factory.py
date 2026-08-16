"""Embedding model factory — config in, model out (Rule 3).

`model_id: hash` selects the deterministic test double; anything else
builds the Sentence Transformers adapter (lazy-loaded, never imported here).
"""

from __future__ import annotations

from app.core.config import EmbeddingSettings
from app.core.exceptions import ConfigurationError
from app.embeddings.base import EmbeddingModel
from app.embeddings.hash_model import HashEmbeddingModel
from app.embeddings.registry import EmbeddingRegistry
from app.embeddings.sentence_transformer_model import SentenceTransformerModel


def build_embedding_model(
    registry: EmbeddingRegistry,
    settings: EmbeddingSettings,
    model_key: str | None = None,
) -> EmbeddingModel:
    key = model_key or settings.default_model
    config = registry.models.get(key)
    if config is None:
        raise ConfigurationError(
            f"Embedding model '{key}' is not defined in {settings.registry_path}"
        )
    if config.model_id == "hash":
        return HashEmbeddingModel(
            dimension=config.dimension,
            model_key=key,
            model_version=config.model_version,
            max_tokens=config.max_tokens,
        )
    return SentenceTransformerModel(config, model_key=key, device=settings.device)