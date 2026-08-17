"""Hybrid failure-boundary tests (ARCHITECTURE §9): timeouts, degrade, strict."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.core.config import RetrievalSettings
from app.core.exceptions import RetrievalUnavailableError
from app.retrieval.base import RetrievalMetadata, RetrievalResult, RetrievedChunk
from app.retrieval.hybrid import HybridRetriever


class StubRetriever:
    def __init__(self, name: str, items: list[RetrievedChunk] | None = None,
                 error: Exception | None = None, delay: float = 0.0) -> None:
        self.name = name
        self._items = items or []
        self._error = error
        self._delay = delay

    async def retrieve(self, query: str, *, top_k: int = 10, filter_=None) -> RetrievalResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return RetrievalResult(
            items=self._items[:top_k],
            metadata=RetrievalMetadata(retriever=self.name, counts={self.name: len(self._items)}),
        )


def _items(n: int, prefix: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
            score=1.0 - i * 0.1, content=f"{prefix}-{i}", sources=(prefix,),
        )
        for i in range(n)
    ]


def _settings(policy: str = "degrade", timeout: float = 1.0) -> RetrievalSettings:
    return RetrievalSettings(
        dense_top_k=10, sparse_top_k=10, final_top_k=5,
        fusion_strategy="rrf", degrade_policy=policy,  # type: ignore[arg-type]
        branch_timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_degrade_policy_serves_surviving_branch() -> None:
    sparse_items = _items(3, "sparse")
    hybrid = HybridRetriever(
        StubRetriever("dense", error=RuntimeError("qdrant down")),
        StubRetriever("sparse", items=sparse_items),
        _settings(policy="degrade"),
    )
    result = await hybrid.retrieve("refund policy")
    assert result.metadata.degraded is True
    assert "dense" in (result.metadata.degraded_reason or "")
    assert len(result.items) == 3


@pytest.mark.asyncio
async def test_strict_policy_raises_on_branch_failure() -> None:
    hybrid = HybridRetriever(
        StubRetriever("dense", error=RuntimeError("boom")),
        StubRetriever("sparse", items=_items(3, "sparse")),
        _settings(policy="strict"),
    )
    with pytest.raises(RetrievalUnavailableError):
        await hybrid.retrieve("refund policy")


@pytest.mark.asyncio
async def test_both_branches_down_always_raises() -> None:
    hybrid = HybridRetriever(
        StubRetriever("dense", error=RuntimeError("a")),
        StubRetriever("sparse", error=TimeoutError("b")),
        _settings(policy="degrade"),
    )
    with pytest.raises(RetrievalUnavailableError):
        await hybrid.retrieve("refund policy")


@pytest.mark.asyncio
async def test_branch_timeout_counts_as_failure() -> None:
    hybrid = HybridRetriever(
        StubRetriever("dense", items=_items(2, "dense"), delay=0.5),
        StubRetriever("sparse", items=_items(2, "sparse")),
        _settings(policy="degrade", timeout=0.05),
    )
    result = await hybrid.retrieve("refund policy")
    assert result.metadata.degraded is True
    assert result.items  # sparse carried the response


@pytest.mark.asyncio
async def test_happy_path_fuses_both_branches() -> None:
    hybrid = HybridRetriever(
        StubRetriever("dense", items=_items(4, "dense")),
        StubRetriever("sparse", items=_items(4, "sparse")),
        _settings(),
    )
    result = await hybrid.retrieve("refund policy")
    assert result.metadata.degraded is False
    assert result.metadata.strategy == "rrf"
    assert result.metadata.counts["dense"] == 4
    assert result.metadata.counts["sparse"] == 4
    assert len(result.items) == 5  # final_top_k
    assert result.metadata.branch_latencies_ms.keys() == {"dense", "sparse"}