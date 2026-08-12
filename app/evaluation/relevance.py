"""Evidence-based relevance adjudication (decision D-029).

Ground truth for chunking experiments cannot be defined at chunk level —
chunks change with every strategy. It is defined at EVIDENCE level instead:
each sample carries verbatim evidence spans from the corpus, and a chunk is
relevant iff an evidence span is contained in it (exact, normalized) or the
token-overlap ratio clears a fixed threshold. Deterministic and strategy-agnostic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.retrieval.bm25 import tokenize

_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_MIN_OVERLAP = 0.7


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def evidence_contained(chunk_text: str, evidence: str) -> bool:
    return normalize(evidence) in normalize(chunk_text)


def token_overlap_ratio(chunk_text: str, evidence: str) -> float:
    """Fraction of unique evidence tokens present in the chunk."""
    evidence_tokens = set(tokenize(evidence))
    if not evidence_tokens:
        return 0.0
    chunk_tokens = set(tokenize(chunk_text))
    return len(evidence_tokens & chunk_tokens) / len(evidence_tokens)


def is_relevant(
    chunk_text: str, evidence: str, min_overlap: float = DEFAULT_MIN_OVERLAP
) -> bool:
    if evidence_contained(chunk_text, evidence):
        return True
    return token_overlap_ratio(chunk_text, evidence) >= min_overlap


def relevant_chunk_indices(
    chunks: Sequence[str],
    evidences: Sequence[str],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> set[int]:
    """Union over evidence spans: any chunk supporting any evidence is gold."""
    relevant: set[int] = set()
    for i, chunk in enumerate(chunks):
        if any(is_relevant(chunk, evidence, min_overlap) for evidence in evidences):
            relevant.add(i)
    return relevant