"""Evaluation dataset schema (Phase 20, decision D-114).

The dataset is a structured artifact with:
- schema_version: for evolution without breaking loaders
- corpus_version: ties questions to a specific corpus snapshot
- questions: gold samples with evidence spans, reference answers, taxonomy labels

Taxonomy (D-113):
- simple_factual: single document, single chunk, straightforward retrieval
- multi_hop: requires combining 2+ chunks or documents
- ambiguous: multiple valid interpretations or missing context
- unanswerable: no evidence in corpus (system must say so, not fabricate)
- temporal_version: requires understanding document versions/dates
- metadata_filter: requires filtering by source_type/tags/department
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceSpan(BaseModel):
    """A verbatim evidence span from the corpus supporting the reference answer."""

    document: str = Field(..., description="Document filename (e.g., 'refund_policy.md')")
    text: str = Field(..., min_length=1, description="Verbatim text from the document")
    section: str | None = Field(None, description="Section heading if applicable")
    page: int | None = Field(None, ge=1, description="Page number if applicable")


class EvaluationQuestion(BaseModel):
    """A gold evaluation sample."""

    id: str = Field(..., pattern=r"^q\d{3}$", description="Unique ID (e.g., 'q001')")
    category: Literal[
        "simple_factual",
        "multi_hop",
        "ambiguous",
        "unanswerable",
        "temporal_version",
        "metadata_filter",
    ] = Field(..., description="Question taxonomy")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Estimated difficulty")
    question: str = Field(..., min_length=1)
    evidence: list[EvidenceSpan] = Field(
        ..., min_length=0, description="Evidence spans (empty for unanswerable)"
    )
    reference_answer: str = Field(
        ..., min_length=1, description="Human-written reference answer"
    )
    expected_sources: list[str] = Field(
        ..., description="Documents that should be retrieved (filenames)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags (e.g., 'refund', 'security', 'enterprise')",
    )

    @field_validator("evidence")
    @classmethod
    def evidence_required_unless_unanswerable(cls, v: list[EvidenceSpan], info) -> list[EvidenceSpan]:
        category = info.data.get("category")
        if category != "unanswerable" and len(v) == 0:
            raise ValueError("Non-unanswerable questions must have at least one evidence span")
        if category == "unanswerable" and len(v) > 0:
            raise ValueError("Unanswerable questions must have empty evidence list")
        return v

    @field_validator("expected_sources")
    @classmethod
    def sources_must_be_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("expected_sources must be non-empty")
        return v


class EvaluationDataset(BaseModel):
    """Top-level dataset structure."""

    schema_version: str = Field(..., description="Schema version (e.g., '1.0')")
    corpus_version: str = Field(..., description="Corpus snapshot version")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    questions: list[EvaluationQuestion] = Field(..., min_length=1)
    metadata: dict[str, str | int] = Field(default_factory=dict)

    @field_validator("questions")
    @classmethod
    def ids_must_be_unique(cls, v: list[EvaluationQuestion]) -> list[EvaluationQuestion]:
        ids = [q.id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return v