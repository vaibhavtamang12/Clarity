"""Context selection — token-budgeted, citation-numbered passage assembly.

Selects the highest-ranked chunks that fit the context budget (greedy in
reranked order), numbers them [1..n], and renders each inside delimited
<passage> tags carrying citation metadata. Passage content is UNTRUSTED
data — the framing is structural groundwork for prompt-injection defense
(D-061, Rule 11; hardened in Phase 25).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.ingestion.tokens import HeuristicTokenCounter, TokenCounter
from app.retrieval.base import RetrievedChunk

_HEADER_TOKEN_OVERHEAD = 12  # conservative allowance for the passage header


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


def _escape_attr(value: str) -> str:
    return value.replace('"', "'")


def render_passage(number: int, chunk: RetrievedChunk) -> str:
    attrs = [f'id="{number}"']
    if chunk.source_uri:
        attrs.append(f'document="{_escape_attr(chunk.source_uri)}"')
    if chunk.page_start is not None:
        page = str(chunk.page_start)
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            page = f"{chunk.page_start}-{chunk.page_end}"
        attrs.append(f'page="{page}"')
    if chunk.section:
        attrs.append(f'section="{_escape_attr(chunk.section)}"')
    return f"<passage {' '.join(attrs)}>\n{chunk.content}\n</passage>"


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
            content_tokens = self._counter.count(item.content)
            total = content_tokens + _HEADER_TOKEN_OVERHEAD
            if used + total > max_tokens:
                if not passages:
                    # Never return empty context when we have something: truncate
                    # the best chunk to fit the budget.
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