"""Factory + collection naming tests (no model download involved)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import EmbeddingSettings
from app.core.exceptions import ConfigurationError
from app.embeddings.factory import build_embedding_model
from app.embeddings.hash_model import HashEmbeddingModel
from app.embeddings.naming import collection_name
from app.embeddings.registry import load_embedding_registry
from app.embeddings.sentence_transformer_model import SentenceTransformerModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry():
    return load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")


def test_factory_builds_hash_model_from_config() -> None:
    model = build_embedding_model(_registry(), EmbeddingSettings(), model_key="hash_64")
    assert isinstance(model, HashEmbeddingModel)
    assert model.dimension == 64
    assert model.model_version == "1"


def test_factory_builds_lazy_sentence_transformer_adapter() -> None:
    # Construction must NOT download or import torch — loading is lazy.
    model = build_embedding_model(_registry(), EmbeddingSettings(), model_key="bge_m3")
    assert isinstance(model, SentenceTransformerModel)
    assert model.dimension == 1024
    assert model.model_id == "BAAI/bge-m3"


def test_factory_defaults_to_configured_model() -> None:
    settings = EmbeddingSettings(default_model="hash_64")
    model = build_embedding_model(_registry(), settings)
    assert model.model_key == "hash_64"


def test_factory_rejects_unknown_key() -> None:
    with pytest.raises(ConfigurationError):
        build_embedding_model(_registry(), EmbeddingSettings(), model_key="ghost_model")


def test_collection_naming_is_model_scoped() -> None:
    assert collection_name("rag", "bge_m3") == "rag__chunks__bge_m3"
    assert collection_name("", "bge_m3") == "chunks__bge_m3"
    assert collection_name("rag", "bge_m3") != collection_name("rag", "multilingual_e5_large")