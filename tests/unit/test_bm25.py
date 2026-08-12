"""Sanity tests for the pure-Python BM25 implementation."""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index, tokenize


DOCS = [
    "The refund policy for enterprise customers allows refunds within 30 days.",
    "Security policy requires quarterly access reviews for all systems.",
    "Enterprise plans include priority support and a dedicated manager.",
]


def test_matching_document_ranks_first() -> None:
    index = BM25Index(DOCS)
    results = index.search("enterprise refund policy", top_k=3)
    assert results
    assert results[0].index == 0
    assert all(r.score > 0 for r in results)


def test_nonmatching_documents_excluded() -> None:
    index = BM25Index(DOCS)
    results = index.search("kubernetes autoscaling", top_k=3)
    assert results == []


def test_more_specific_match_outranks_partial() -> None:
    index = BM25Index(DOCS)
    results = index.search("quarterly access reviews security", top_k=3)
    assert results[0].index == 1


def test_tokenize_is_deterministic_and_lowercase() -> None:
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_search_is_deterministic() -> None:
    index = BM25Index(DOCS)
    first = [(r.index, r.score) for r in index.search("enterprise", top_k=3)]
    second = [(r.index, r.score) for r in index.search("enterprise", top_k=3)]
    assert first == second