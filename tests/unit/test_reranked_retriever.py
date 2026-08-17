"""RerankedRetriever: candidate cut, score replacement, provenance, fail-soft."""

from __future__ import annotations

import uuid

import pytest

from app.reranking.hash_reranker import HashReranker
from app.reranking.retriever import RerankedRetriever
from app.retrieval.base import RetrievalMetadata, RetrievalResult, RetrievedChunk, Retriever


class StaticRetriever:
    name = "static"

    def __init__(self, items: list[RetrievedChunk]) -> None:
        self.items = items
        self.requested_top_k: list[int] = []

    async def retrieve(self, query: str, *, top_k: int = 10, filter_=None) -> RetrievalResult:
        self.requested_top_k.append(top_k)
        return RetrievalResult(
            items=self.items[:top_k],
            metadata=RetrievalMetadata(
                retriever="hybrid", strategy="rrf",
                counts={"dense": len(self.items), "sparse": len(self.items)},
                branch_latencies_ms={"dense": 1.0, "sparse": 1.0},
            ),
        )


class BrokenReranker:
    model_key = "broken"
    model_id = "broken"

    def rerank(self, query: str, items):
        raise RuntimeError("model crashed")


def _items(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
            score=1.0 - i * 0.01, content=f"enterprise refund policy clause number {i}",
            sources=("dense", "sparse"), dense_score=0.8, sparse_score=2.0,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_retrieves_wide_and_keeps_narrow() -> None:
    base = StaticRetriever(_items(25))
    retriever = RerankedRetriever(base, HashReranker(), candidates=20, top_n=5)

    result = await retriever.retrieve("enterprise refund policy", top_k=5)

    assert base.requested_top_k == [20]          # retrieved wide...
    assert len(result.items) == 5                # ...kept narrow
    assert result.metadata.counts["rerank_candidates"] == 20
    assert result.metadata.counts["rerank_kept"] == 5
    assert result.metadata.strategy == "rrf+rerank"
    assert "rerank" in result.metadata.branch_latencies_ms


@pytest.mark.asyncio
async def test_scores_replaced_and_branch_scores_preserved() -> None:
    base = StaticRetriever(_items(10))
    retriever = RerankedRetriever(base, HashReranker(), candidates=10, top_n=10)

    result = await retriever.retrieve("enterprise refund policy", top_k=10)
    for item in result.items:
        assert item.rerank_score is not None
        assert item.score == item.rerank_score
        assert item.dense_score == 0.8           # provenance survives reranking
        assert item.sparse_score == 2.0


@pytest.mark.asyncio
async def test_reranking_changes_ordering() -> None:
    good = RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
        score=0.10, content="enterprise refund window is 30 days", sources=("sparse",),
    )
    weak = RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
        score=0.99, content="unrelated cafeteria menu", sources=("dense",),
    )
    retriever = RerankedRetriever(StaticRetriever([weak, good]), HashReranker(), candidates=10, top_n=2)

    result = await retriever.retrieve("enterprise refund window", top_k=2)
    assert result.items[0].chunk_id == good.chunk_id   # low base rank, high rerank score


@pytest.mark.asyncio
async def test_reranker_failure_degrades_to_base_ordering() -> None:
    base_items = _items(8)
    retriever = RerankedRetriever(StaticRetriever(base_items), BrokenReranker(), candidates=10, top_n=4)

    result = await retriever.retrieve("enterprise refund policy", top_k=4)
    assert result.metadata.degraded is True
    assert "rerank failed" in (result.metadata.degraded_reason or "")
    assert [i.chunk_id for i in result.items] == [i.chunk_id for i in base_items[:4]]


@pytest.mark.asyncio
async def test_disabled_reranker_passes_through() -> None:
    base = StaticRetriever(_items(6))
    retriever = RerankedRetriever(base, HashReranker(), candidates=20, top_n=5, enabled=False)
    result = await retriever.retrieve("q", top_k=3)
    assert len(result.items) == 3
    assert base.requested_top_k == [3]            # no wide retrieval when disabled


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        RerankedRetriever(StaticRetriever([]), HashReranker(), candidates=5, top_n=10)