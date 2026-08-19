"""Context selection — token-budgeted, citation-numbered passage assembly.

Phase 25 hardening: passage content and attribute values are XML-escaped
before entering the prompt. A document containing '</passage>' or fake tags
can no longer break out of the passage structure — the primary structural
defense against prompt injection (D-137). Suspicious payloads are flagged as
security events (alerting, never blocking — D-138).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ingestion.tokens import HeuristicTokenCounter, TokenCounter
from app.retrieval.base import RetrievedChunk
from app.security.audit import record_security_event
from app.security.injection import (
    escape_attribute_value,
    escape_passage_content,
    scan_for_injection,
)

_HEADER_TOKEN_OVERHEAD = 12


@dataclass(frozen=True)
class Passage:
    number: int
    chunk: RetrievedChunk


@dataclass(frozen=True)
class ContextPack:
    passages: tuple[Passage, ...]
    text: str
    used_tokens: int
    dropped_count: int

    def passage_by_number(self, number: int) -> Passage | None:
        return next((p for p in self.passages if p.number == number), None)

    @property
    def chunk_ids(self) -> set:
        return {p.chunk.chunk_id for p in self.passages}


def render_passage(number: int, chunk: RetrievedChunk) -> str:
    attrs = [f'id="{number}"']
    if chunk.source_uri:
        attrs.append(f'document="{escape_attribute_value(chunk.source_uri)}"')
    if chunk.page_start is not None:
        page = str(chunk.page_start)
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            page = f"{chunk.page_start}-{chunk.page_end}"
        attrs.append(f'page="{page}"')
    if chunk.section:
        attrs.append(f'section="{escape_attribute_value(chunk.section)}"')
    # PRIMARY injection defense: escaped content can never close or forge tags.
    safe_content = escape_passage_content(chunk.content)
    return f"<passage {' '.join(attrs)}>\n{safe_content}\n</passage>"


class ContextBuilder:
    def __init__(
        self, token_counter: TokenCounter | None = None, max_passages: int = 8
    ) -> None:
        self._counter = token_counter or HeuristicTokenCounter()
        self._max_passages = max_passages

    def build(self, items: Sequence[RetrievedChunk], max_tokens: int) -> ContextPack:
        passages: list[Passage] = []
        used = 0
        dropped = 0

        for item in items:
            if len(passages) >= self._max_passages:
                dropped += 1
                continue

            # Defense-in-depth: flag known injection signatures (alert only).
            scan = scan_for_injection(item.content)
            if scan.suspicious:
                record_security_event(
                    "injection_pattern_detected",
                    patterns=",".join(scan.patterns_matched),
                    source=item.source_uri or "unknown",
                )

            content_tokens = self._counter.count(item.content)
            total = content_tokens + _HEADER_TOKEN_OVERHEAD
            if used + total > max_tokens:
                if not passages:
                    budget_chars = max(0, (max_tokens - _HEADER_TOKEN_OVERHEAD)) * 4
                    item = RetrievedChunk(
                        chunk_id=item.chunk_id, document_id=item.document_id,
                        version_id=item.version_id, score=item.score,
                        content=item.content[:budget_chars], sources=item.sources,
                        dense_score=item.dense_score, sparse_score=item.sparse_score,
                        rerank_score=item.rerank_score, section=item.section,
                        source_uri=item.source_uri, title=item.title,
                        page_start=item.page_start, page_end=item.page_end,
                    )
                    total = self._counter.count(item.content) + _HEADER_TOKEN_OVERHEAD
                else:
                    dropped += len(items) - items.index(item)
                    break
            passages.append(Passage(number=len(passages) + 1, chunk=item))
            used += total

        text = ""
        if passages:
            text = "<context>\n" + "\n\n".join(
                render_passage(p.number, p.chunk) for p in passages
            ) + "\n</context>"
        return ContextPack(
            passages=tuple(passages), text=text, used_tokens=used, dropped_count=dropped
        )