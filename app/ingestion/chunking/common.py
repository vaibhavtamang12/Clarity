"""Text splitters shared by all chunking strategies."""

from __future__ import annotations

import re

from app.ingestion.tokens import TokenCounter

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_RE.split(text) if s.strip()]


def hard_split(text: str, max_tokens: int, counter: TokenCounter) -> list[str]:
    """Last-resort splitter: word boundaries first, character cuts for monsters."""
    if counter.count(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if counter.count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if counter.count(word) > max_tokens:
            step = max(1, max_tokens * 4)
            pieces.extend(word[i : i + step] for i in range(0, len(word), step))
            current = ""
        else:
            current = word
    if current:
        pieces.append(current)
    return pieces


def recursive_split(
    text: str,
    max_tokens: int,
    counter: TokenCounter,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[str]:
    """Classic recursive character split: try the coarsest separator that
    exists, merge parts up to budget, recurse into still-oversized parts
    with finer separators."""
    if counter.count(text) <= max_tokens:
        return [text]

    for i, sep in enumerate(separators):
        if not sep or sep not in text:
            continue
        parts = text.split(sep)
        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if counter.count(candidate) <= max_tokens:
                current = candidate
                continue
            if current:
                pieces.append(current)
                current = ""
            if counter.count(part) > max_tokens:
                pieces.extend(recursive_split(part, max_tokens, counter, separators[i + 1 :]))
            else:
                current = part
        if current:
            pieces.append(current)
        return [p for p in pieces if p.strip()]

    return hard_split(text, max_tokens, counter)