"""Reranker model tests: hash double + cross-encoder wrapper (no torch needed)."""

from __future__ import annotations

import uuid

import pytest

from app.core.exceptions import RerankerUnavailableError
from app.reranking.cross_encoder_model import CrossEncoderReranker
from app.reranking.hash_reranker import HashReranker
from app.reranking.registry import RerankerModelConfig
from app.retrieval.base import RetrievedChunk


def _item(content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
        score=score, content=content, sources=("dense",),
    )


def test_hash_reranker_orders_by_overlap() -> None:
    reranker = HashReranker()
    items = [
        _item("passwords must have fourteen characters minimum"),
        _item("enterprise refund window is 30 days from invoice date"),
    ]
    scores = reranker.rerank("enterprise refund window", items)
    assert scores[1] > scores[0]
    assert scores[1] > 0.5  # most query tokens matched


def test_hash_reranker_is_deterministic() -> None:
    reranker = HashReranker()
    items = [_item("refund policy for enterprise customers")]
    assert reranker.rerank("refund policy", items) == reranker.rerank("refund policy", items)


def test_hash_reranker_empty_inputs() -> None:
    assert HashReranker().rerank("anything", []) == []
    assert HashReranker().rerank("", [_item("text")]) == [0.0]


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        self.calls.append(list(pairs))
        return self._scores[: len(pairs)]


def _ce_reranker(fake: FakeCrossEncoder) -> CrossEncoderReranker:
    config = RerankerModelConfig(model_id="fake/model", batch_size=8)
    return CrossEncoderReranker(config, model_key="fake", loader=lambda _m, _d: fake)


def test_cross_encoder_wrapper_lazy_loads_and_scores() -> None:
    fake = FakeCrossEncoder(scores=[0.2, 0.9])
    reranker = _ce_reranker(fake)
    assert fake.calls == []  # nothing loaded at construction

    scores = reranker.rerank("q", [_item("a"), _item("b")])
    assert scores == [0.2, 0.9]
    assert fake.calls[0][0] == ("q", "a")


def test_cross_encoder_wrapper_normalizes_failures() -> None:
    class ExplodingCE:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("gpu fell over")

    reranker = CrossEncoderReranker(
        RerankerModelConfig(model_id="x/y"), model_key="x", loader=lambda _m, _d: ExplodingCE()
    )
    with pytest.raises(RerankerUnavailableError):
        reranker.rerank("q", [_item("a")])


def test_cross_encoder_empty_items_short_circuits() -> None:
    fake = FakeCrossEncoder(scores=[])
    assert _ce_reranker(fake).rerank("q", []) == []
    assert fake.calls == []