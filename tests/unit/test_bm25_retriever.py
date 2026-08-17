"""BM25Retriever protocol behavior."""

from __future__ import annotations

import uuid

import pytest

from app.repositories.vector.base import VectorFilter
from app.retrieval.bm25 import BM25Index
from app.retrieval.sparse import BM25Retriever, SparseChunkRecord


def _build():
    texts = [
        "Enterprise refunds are possible within 30 days.",
        "Passwords must have at least 14 characters.",
        "The API rate limit is 60 requests per minute.",
    ]
    records = [
        SparseChunkRecord(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(), content=text
        )
        for text in texts
    ]
    return BM25Retriever(BM25Index(texts), records), records


@pytest.mark.asyncio
async def test_retriever_protocol_and_ranking() -> None:
    retriever, records = _build()
    result = await retriever.retrieve("enterprise refund window", top_k=3)
    assert result.metadata.retriever == "sparse_bm25"
    assert result.items[0].chunk_id == records[0].chunk_id
    assert result.items[0].sources == ("sparse",)
    assert result.items[0].sparse_score == result.items[0].score


@pytest.mark.asyncio
async def test_filters_are_rejected_explicitly() -> None:
    retriever, _ = _build()
    with pytest.raises(ValueError):
        await retriever.retrieve("x", filter_=VectorFilter(document_id=uuid.uuid4()))


def test_index_record_alignment_is_enforced() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(BM25Index(["only one"]), [])