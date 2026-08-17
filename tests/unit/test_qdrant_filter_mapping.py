"""The VectorFilter → Qdrant mapping is pure and server-free testable (D-042)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from qdrant_client import models as qmodels

from app.repositories.vector.base import VectorFilter
from app.repositories.vector.qdrant_repository import build_qdrant_filter


def test_empty_filter_maps_to_none() -> None:
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(VectorFilter()) is None


def test_all_conditions_are_generated() -> None:
    now = datetime.now(timezone.utc)
    qfilter = build_qdrant_filter(
        VectorFilter(
            document_id=uuid.uuid4(),
            version_id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            source_type="pdf",
            document_type="policy",
            department="legal",
            tags=("a", "b"),
            is_active_version=True,
            created_after=now,
            created_before=now,
        )
    )
    assert qfilter is not None
    assert isinstance(qfilter, qmodels.Filter)
    assert len(qfilter.must) == 10  # one condition per filter dimension


def test_uuid_fields_are_serialized_as_strings() -> None:
    doc_id = uuid.uuid4()
    qfilter = build_qdrant_filter(VectorFilter(document_id=doc_id))
    condition = qfilter.must[0]
    assert isinstance(condition, qmodels.FieldCondition)
    assert condition.key == "document_id"
    assert condition.match.value == str(doc_id)


def test_tags_use_match_any() -> None:
    qfilter = build_qdrant_filter(VectorFilter(tags=("legal", "billing")))
    condition = qfilter.must[0]
    assert isinstance(condition.match, qmodels.MatchAny)
    assert set(condition.match.any) == {"legal", "billing"}