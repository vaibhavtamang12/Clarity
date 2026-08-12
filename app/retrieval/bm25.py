"""Pure-Python Okapi BM25 — the canonical lexical ranker for evaluation harnesses.

The production sparse path (Phase 8) is PostgreSQL tsvector (decision D-010).
This implementation exists so experiments and tests have a deterministic,
infrastructure-free lexical retriever. Phase 8 benchmarks the two against
each other rather than assuming equivalence.
"""

from __future__ import annotations

import math
import re
from collections import Counter, Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — simple, deterministic, language-agnostic."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ScoredDocument:
    index: int
    score: float


class BM25Index:
    """Okapi BM25 with the standard non-negative IDF variant.

    score(D, Q) = Σ_{t ∈ Q} idf(t) · tf(t,D)·(k1+1) / (tf(t,D) + k1·(1 − b + b·|D|/avgdl))
    idf(t)      = ln((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens = [tokenize(doc) for doc in documents]
        self._term_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._lengths = [len(tokens) for tokens in self._doc_tokens]
        self._doc_count = len(documents)
        self._avgdl = sum(self._lengths) / self._doc_count if self._doc_count else 0.0

        doc_freq: Counter[str] = Counter()
        for freq in self._term_freqs:
            doc_freq.update(freq.keys())
        self._idf = {
            term: math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    @property
    def document_count(self) -> int:
        return self._doc_count

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        query_terms = tokenize(query)
        scored: list[tuple[int, float]] = []
        for i, freq in enumerate(self._term_freqs):
            score = 0.0
            length = self._lengths[i]
            for term in query_terms:
                tf = freq.get(term, 0)
                if tf == 0 or self._avgdl == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self._avgdl)
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [ScoredDocument(index=i, score=s) for i, s in scored[:top_k]]