"""EmbeddingService — the operational wrapper (decision D-034).

Responsibilities, in order:
1. Batched embedding with input-order preservation.
2. Cache lookup before inference; cache write after (fail-open both ways).
3. Retries with exponential backoff for transient inference failures.
   Non-transient failures (missing deps, bad config) propagate immediately.
4. Dimension validation — a model/config mismatch is caught here, not in
   the vector index.
5. Structured latency logging (operational metadata only — Rule 10).

Inference runs via asyncio.to_thread so CPU-bound model work never blocks
the event loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from app.core.exceptions import EmbeddingUnavailableError
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingModel
from app.embeddings.caching import (
    CacheStore,
    deserialize_vector,
    embedding_cache_key,
    serialize_vector,
)

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        model: EmbeddingModel,
        cache: CacheStore | None = None,
        cache_ttl_seconds: int = 86400,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
    ) -> None:
        self._model = model
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_seconds

    # ---------------------------------------------------------------- identity
    @property
    def model_key(self) -> str:
        return self._model.model_key

    @property
    def model_version(self) -> str:
        return self._model.model_version

    @property
    def dimension(self) -> int:
        return self._model.dimension

    # ------------------------------------------------------------------- public
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        results: dict[int, list[float]] = {}
        keys = [
            embedding_cache_key(self._model.model_key, self._model.model_version, text)
            for text in texts
        ]

        # ---- cache lookup (fail-open) -------------------------------------
        if self._cache is not None:
            try:
                cached = await self._cache.get_many(keys)
                for i, key in enumerate(keys):
                    if key in cached:
                        results[i] = deserialize_vector(cached[key])
            except Exception as exc:  # noqa: BLE001 — cache outage ≠ embedding outage
                logger.warning("embedding_cache_lookup_failed", error=type(exc).__name__)
                results = {}

        # ---- inference for the misses --------------------------------------
        miss_indices = [i for i in range(len(texts)) if i not in results]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            vectors = await self._embed_with_retry(miss_texts)
            for i, vector in zip(miss_indices, vectors):
                results[i] = vector
            await self._store(miss_indices, keys, results)

        return [results[i] for i in range(len(texts))]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    # ------------------------------------------------------------------ internals
    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                start = time.perf_counter()
                vectors = await asyncio.to_thread(self._model.embed_documents, texts)
                latency_ms = (time.perf_counter() - start) * 1000
                self._validate(vectors)
                logger.info(
                    "embedding_batch_completed",
                    model=self._model.model_key,
                    count=len(texts),
                    attempt=attempt,
                    latency_ms=round(latency_ms, 2),
                )
                return vectors
            except EmbeddingUnavailableError:
                raise  # configuration/dependency errors are not transient
            except Exception as exc:  # noqa: BLE001 — transient inference failures retry
                last_error = exc
                logger.warning(
                    "embedding_batch_failed",
                    model=self._model.model_key,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if attempt < attempts:
                    await asyncio.sleep(self._retry_base_delay * (2 ** (attempt - 1)))
        raise EmbeddingUnavailableError(
            f"Embedding failed after {attempts} attempts: {type(last_error).__name__}"
        ) from last_error

    def _validate(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self._model.dimension:
                raise EmbeddingUnavailableError(
                    f"Model returned dimension {len(vector)}, "
                    f"expected {self._model.dimension} — config/model mismatch"
                )

    async def _store(
        self, indices: list[int], keys: list[str], results: dict[int, list[float]]
    ) -> None:
        if self._cache is None:
            return
        for i in indices:
            try:
                await self._cache.set(keys[i], serialize_vector(results[i]), self._cache_ttl)
            except Exception as exc:  # noqa: BLE001 — fail-open on write too
                logger.warning("embedding_cache_write_failed", error=type(exc).__name__)
                return