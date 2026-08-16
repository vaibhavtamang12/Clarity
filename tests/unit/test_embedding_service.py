"""Tests for batching, caching (incl. fail-open), retries, and validation."""

from __future__ import annotations

import pytest

from app.core.exceptions import EmbeddingUnavailableError
from app.embeddings.caching import InMemoryCacheStore, embedding_cache_key, serialize_vector
from app.embeddings.service import EmbeddingService


class CountingModel:
    model_key = "test-model"
    model_id = "test"
    model_version = "9"
    dimension = 4
    max_tokens = 512

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text))] * 4 for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class RaisingStore:
    async def get_many(self, keys):
        raise ConnectionError("redis down")

    async def set(self, key, value, ttl_seconds):
        raise ConnectionError("redis down")


class FlakyModel(CountingModel):
    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self.fail_times = fail_times

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("transient inference failure")
        return [[1.0] * 4 for _ in texts]


class WrongDimModel(CountingModel):
    def embed_documents(self, texts):
        return [[1.0, 2.0] for _ in texts]  # dimension 2 ≠ 4


def _service(model, cache=None, **kwargs) -> EmbeddingService:
    return EmbeddingService(
        model, cache=cache, cache_ttl_seconds=60, max_retries=2, retry_base_delay_seconds=0.0, **kwargs
    )


@pytest.mark.asyncio
async def test_order_preserved_with_partial_cache_hits() -> None:
    model = CountingModel()
    store = InMemoryCacheStore()
    # Pre-seal the cache for the second text only.
    key = embedding_cache_key(model.model_key, model.model_version, "cached text")
    await store.set(key, serialize_vector([9.0, 9.0, 9.0, 9.0]), 60)

    service = _service(model, cache=store)
    vectors = await service.embed_documents(["fresh a", "cached text", "fresh b"])

    assert vectors[1] == [9.0, 9.0, 9.0, 9.0]     # served from cache
    assert vectors[0] == [7.0] * 4                 # len("fresh a")
    assert vectors[2] == [7.0] * 4
    assert model.calls == [["fresh a", "fresh b"]]  # only misses embedded


@pytest.mark.asyncio
async def test_cache_write_populates_store() -> None:
    model = CountingModel()
    store = InMemoryCacheStore()
    service = _service(model, cache=store)
    await service.embed_documents(["hello"])
    assert store.set_calls == 1
    # Second call is fully cached — no new inference.
    await service.embed_documents(["hello"])
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_cache_outage_fails_open() -> None:
    model = CountingModel()
    service = _service(model, cache=RaisingStore())
    vectors = await service.embed_documents(["a", "bb"])
    assert vectors == [[1.0] * 4, [2.0] * 4]  # correctness intact despite Redis outage
    assert model.calls == [["a", "bb"]]


@pytest.mark.asyncio
async def test_transient_failure_recovers_via_retry() -> None:
    model = FlakyModel(fail_times=2)
    service = _service(model)
    vectors = await service.embed_documents(["x"])
    assert vectors == [[1.0] * 4]
    assert len(model.calls) == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_exhausted_retries_raise_typed_error() -> None:
    model = FlakyModel(fail_times=99)
    service = _service(model)
    with pytest.raises(EmbeddingUnavailableError):
        await service.embed_documents(["x"])
    assert len(model.calls) == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_dimension_mismatch_detected() -> None:
    with pytest.raises(EmbeddingUnavailableError):
        await _service(WrongDimModel()).embed_documents(["x"])


@pytest.mark.asyncio
async def test_empty_input_short_circuits() -> None:
    model = CountingModel()
    assert await _service(model).embed_documents([]) == []
    assert model.calls == []


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector() -> None:
    model = CountingModel()
    service = _service(model, cache=InMemoryCacheStore())
    vector = await service.embed_query("abc")
    assert vector == [3.0] * 4