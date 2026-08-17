"""Hand-computed tests for both fusion engines."""

from __future__ import annotations

import uuid

from app.retrieval.base import RetrievedChunk
from app.retrieval.fusion import fuse


def _chunk(content: str, score: float, sources: tuple[str, ...] = ("dense",)) -> RetrievedChunk:
    cid = uuid.uuid4()
    return RetrievedChunk(
        chunk_id=cid,
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        score=score,
        content=content,
        sources=sources,
        dense_score=score if "dense" in sources else None,
        sparse_score=score if "sparse" in sources else None,
    )


def test_rrf_hand_computed_ordering() -> None:
    c1, c2, c3 = _chunk("c1", 0.9), _chunk("c2", 0.8), _chunk("c3", 0.7)
    dense = [c1, c2]
    sparse = [
        RetrievedChunk(chunk_id=c2.chunk_id, document_id=c2.document_id, version_id=c2.version_id,
                       score=5.0, content="c2", sources=("sparse",), sparse_score=5.0),
        RetrievedChunk(chunk_id=c3.chunk_id, document_id=c3.document_id, version_id=c3.version_id,
                       score=4.0, content="c3", sources=("sparse",), sparse_score=4.0),
    ]
    fused = fuse({"dense": dense, "sparse": sparse}, "rrf", rrf_k=60)

    # c2: 1/62 + 1/61  >  c1: 1/61  >  c3: 1/62
    assert [item.content for item in fused] == ["c2", "c1", "c3"]
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9
    assert set(fused[0].sources) == {"dense", "sparse"}
    assert fused[0].dense_score == 0.8 and fused[0].sparse_score == 5.0


def test_weighted_fusion_normalizes_scales() -> None:
    a = _chunk("a", 0.9)                       # dense-only, top dense score
    b = _chunk("b", 0.5)                       # appears in both lists
    c_sparse = RetrievedChunk(
        chunk_id=b.chunk_id, document_id=b.document_id, version_id=b.version_id,
        score=2.0, content="b", sources=("sparse",), sparse_score=2.0,
    )
    c = RetrievedChunk(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), version_id=uuid.uuid4(),
                       score=1.0, content="c", sources=("sparse",), sparse_score=1.0)

    fused = fuse(
        {"dense": [a, b], "sparse": [c_sparse, c]},
        "weighted",
        weights={"dense": 0.7, "sparse": 0.3},
    )
    # dense minmax: a=1.0, b=0.0 · sparse minmax: b=1.0, c=0.0
    # a=0.7 · b=0.3 · c=0.0
    assert [item.content for item in fused] == ["a", "b", "c"]
    assert abs(fused[0].score - 0.7) < 1e-9
    assert abs(fused[1].score - 0.3) < 1e-9
    assert set(fused[1].sources) == {"dense", "sparse"}


def test_duplicate_chunks_merge_once() -> None:
    shared = _chunk("shared", 0.9)
    same_in_sparse = RetrievedChunk(
        chunk_id=shared.chunk_id, document_id=shared.document_id, version_id=shared.version_id,
        score=3.0, content="shared", sources=("sparse",), sparse_score=3.0,
    )
    fused = fuse({"dense": [shared], "sparse": [same_in_sparse]}, "rrf")
    assert len(fused) == 1
    assert set(fused[0].sources) == {"dense", "sparse"}


def test_empty_lists_and_unknown_strategy() -> None:
    assert fuse({"dense": [], "sparse": []}, "rrf") == []
    try:
        fuse({}, "quantum")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError:
        pass