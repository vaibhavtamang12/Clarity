"""Citation resolution — passage numbers → verifiable chunk citations.

Everything unresolvable is dropped and counted (never silently kept):
out-of-range passage numbers, duplicates, anything beyond the citation cap.
Claim-level verification against chunk text is Phase 12's engine; Phase 11
guarantees the structural integrity of every citation that survives.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from collections.abc import Sequence

from app.generation.context import ContextPack
from app.generation.domain import Citation, LLMCitationRef


@dataclass(frozen=True)
class CitationResolution:
    citations: list[Citation]
    dropped_count: int
    used_passage_numbers: frozenset[int]


def resolve_citations(
    refs: Sequence[LLMCitationRef], pack: ContextPack, max_citations: int
) -> CitationResolution:
    resolved: list[Citation] = []
    seen_chunks: set[uuid.UUID] = set()
    used_numbers: set[int] = set()
    dropped = 0

    for ref in refs:
        passage = pack.passage_by_number(ref.passage)
        if passage is None:
            dropped += 1                     # hallucinated passage number
            continue
        chunk = passage.chunk
        if chunk.chunk_id in seen_chunks:
            dropped += 1                     # duplicate citation for same chunk
            continue
        if len(resolved) >= max_citations:
            dropped += 1                     # citation cap
            continue
        seen_chunks.add(chunk.chunk_id)
        used_numbers.add(ref.passage)
        resolved.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                passage=ref.passage,
                claim=ref.claim.strip()[:300],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                source_uri=chunk.source_uri,
                title=chunk.title,
            )
        )

    return CitationResolution(
        citations=resolved, dropped_count=dropped, used_passage_numbers=frozenset(used_numbers)
    )