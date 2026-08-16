"""Tests for the deterministic hash embedding test double."""

from __future__ import annotations

import math

from app.embeddings.hash_model import HashEmbeddingModel


def test_deterministic_across_instances() -> None:
    a = HashEmbeddingModel(dimension=64).embed_query("refund policy for enterprise")
    b = HashEmbeddingModel(dimension=64).embed_query("refund policy for enterprise")
    assert a == b


def test_dimension_and_normalization() -> None:
    model = HashEmbeddingModel(dimension=32)
    vector = model.embed_query("the quick brown fox")
    assert len(vector) == 32
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9


def test_different_texts_differ_same_texts_match() -> None:
    model = HashEmbeddingModel(dimension=64)
    v1 = model.embed_query("refund policy")
    v2 = model.embed_query("password requirements")
    v3 = model.embed_query("refund policy")
    assert v1 != v2
    assert v1 == v3


def test_shared_tokens_produce_positive_similarity() -> None:
    model = HashEmbeddingModel(dimension=128)
    v1 = model.embed_query("enterprise refund window is 30 days")
    v2 = model.embed_query("enterprise refund approval takes 5 days")
    dot = sum(a * b for a, b in zip(v1, v2))
    assert dot > 0


def test_empty_text_yields_zero_vector() -> None:
    model = HashEmbeddingModel(dimension=16)
    assert model.embed_query("") == [0.0] * 16


def test_embed_documents_preserves_order() -> None:
    model = HashEmbeddingModel(dimension=16)
    texts = ["alpha", "beta", "alpha"]
    vectors = model.embed_documents(texts)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]