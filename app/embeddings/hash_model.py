"""Deterministic feature-hashing embedder — first-class test double (D-036).

Vectors are signed feature hashes of tokens (the classic hashing trick),
L2-normalized. Texts sharing tokens get non-zero dot products, which is
enough to exercise every pipeline stage end-to-end — CI, local dev, and
integration tests run without downloading a 2 GB model.

It is NOT a quality model: production retrieval quality comes from the
configured Sentence Transformers model (bge-m3 by default). The swap is a
config change, nothing else.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.retrieval.bm25 import tokenize

DEFAULT_DIMENSION = 64


class HashEmbeddingModel:
    model_id = "hash"

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        model_key: str = "hash_64",
        model_version: str = "1",
        max_tokens: int = 8192,
    ) -> None:
        self.dimension = dimension
        self.model_key = model_key
        self.model_version = model_version
        self.max_tokens = max_tokens

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector  # empty text → zero vector (documented behavior)
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)