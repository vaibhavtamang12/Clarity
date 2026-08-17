"""Reranker factory — config in, reranker out (Rule 3)."""

from __future__ import annotations

from app.core.config import RerankerSettings
from app.core.exceptions import ConfigurationError
from app.reranking.base import Reranker
from app.reranking.cross_encoder_model import CrossEncoderReranker
from app.reranking.hash_reranker import HashReranker
from app.reranking.registry import RerankerRegistry


def build_reranker(
    registry: RerankerRegistry,
    settings: RerankerSettings,
    model_key: str | None = None,
) -> Reranker:
    key = model_key or settings.model
    config = registry.models.get(key)
    if config is None:
        raise ConfigurationError(
            f"Reranker '{key}' is not defined in configs/rerankers.yaml"
        )
    if config.model_id == "hash":
        return HashReranker(model_key=key)
    return CrossEncoderReranker(
        config, model_key=key, device=settings.device, loader=None
    )