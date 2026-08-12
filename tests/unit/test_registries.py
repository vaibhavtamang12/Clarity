"""Tests for the embedding + chunking config registries.

These double as validation that the shipped YAML configs are well-formed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import ConfigurationError
from app.embeddings.registry import load_embedding_registry
from app.ingestion.chunking_registry import load_chunking_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


def test_embedding_registry_loads() -> None:
    registry = load_embedding_registry(CONFIGS / "embeddings.yaml")
    assert registry.default in registry.models
    default_model = registry.models[registry.default]
    assert default_model.dimension > 0
    assert default_model.max_tokens > 0


def test_embedding_registry_missing_file() -> None:
    with pytest.raises(ConfigurationError):
        load_embedding_registry(CONFIGS / "does-not-exist.yaml")


def test_embedding_registry_bad_default(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "default: ghost\n"
        "models:\n"
        "  bge:\n"
        "    model_id: x\n"
        "    dimension: 4\n"
        "    max_tokens: 8\n"
    )
    with pytest.raises(ConfigurationError):
        load_embedding_registry(p)


def test_chunking_registry_loads() -> None:
    registry = load_chunking_registry(CONFIGS / "chunking.yaml")
    assert registry.default in registry.strategies
    default_strategy = registry.strategies[registry.default]
    assert 0 <= default_strategy.overlap_tokens < default_strategy.target_tokens
    assert default_strategy.min_tokens <= default_strategy.target_tokens
    assert default_strategy.max_tokens >= default_strategy.target_tokens