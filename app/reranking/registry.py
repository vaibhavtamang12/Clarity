# app/reranking/registry.py
"""Reranker registry — mirrors the embedding registry (Rule 3, D-051).

The active reranker is configuration: RERANKER__MODEL selects a key here.
Profiles:
- bge_reranker_v2_m3 — production default (ADR-004), multilingual
- ms_marco_mini      — fast English-only profile for latency-sensitive runs
- hash               — deterministic CI/dev double, never the production default
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.exceptions import ConfigurationError
from app.utils.yaml import read_yaml

DEFAULT_REGISTRY_PATH = Path("configs/rerankers.yaml")


class RerankerModelConfig(BaseModel):
    model_id: str
    batch_size: int = Field(default=16, gt=0)
    description: str = ""


class RerankerRegistry(BaseModel):
    default: str
    models: dict[str, RerankerModelConfig]


def load_reranker_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> RerankerRegistry:
    raw = read_yaml(path)
    try:
        registry = RerankerRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid reranker registry at {path}: {exc}") from exc
    if registry.default not in registry.models:
        raise ConfigurationError(
            f"Reranker registry default '{registry.default}' is not defined in models"
        )
    return registry