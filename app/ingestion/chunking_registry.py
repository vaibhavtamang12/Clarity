"""Chunking strategy registry (ADR-003).

Strategies are configuration; the Phase 4 pipeline consumes them through a
common Chunker interface, and Phase 5 benchmarks them against each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.exceptions import ConfigurationError
from app.utils.yaml import read_yaml

DEFAULT_REGISTRY_PATH = Path("configs/chunking.yaml")

StrategyType = Literal["fixed", "recursive", "sentence", "structure_aware"]


class ChunkingStrategyConfig(BaseModel):
    type: StrategyType
    target_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    min_tokens: int = Field(gt=0, default=32)
    max_tokens: int = Field(gt=0, default=2048)

    @model_validator(mode="after")
    def _consistent(self) -> "ChunkingStrategyConfig":
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if self.min_tokens > self.target_tokens:
            raise ValueError("min_tokens cannot exceed target_tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens cannot be below target_tokens")
        return self


class ChunkingRegistry(BaseModel):
    default: str
    strategies: dict[str, ChunkingStrategyConfig]


def load_chunking_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> ChunkingRegistry:
    raw = read_yaml(path)
    try:
        registry = ChunkingRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid chunking registry at {path}: {exc}") from exc
    if registry.default not in registry.strategies:
        raise ConfigurationError(
            f"Chunking registry default '{registry.default}' is not defined in strategies"
        )
    return registry