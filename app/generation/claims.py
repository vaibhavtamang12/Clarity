"""Claim extraction from generated answers.

Claim unit = one sentence of the answer together with the passage markers it
carries. Rule-based and deterministic (Rule 6): the LLM already declares its
claims in the structured output (citations[].claim); sentence extraction is
the ANSWER-side view used to cross-check what the model actually wrote.

Markers like [1] / [2][3] are parsed, associated per sentence, and stripped
for matching — the matcher compares semantics, not bracket syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MARKER_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_CLAIM_CHARS = 3


@dataclass(frozen=True)
class ExtractedClaim:
    text: str                  # marker-stripped sentence (what gets matched)
    raw: str                   # original sentence including markers
    markers: tuple[int, ...]   # passage numbers this sentence cites


def strip_markers(text: str) -> str:
    return MARKER_RE.sub("", text)


def extract_markers(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(m) for m in MARKER_RE.findall(text)}))


def extract_claims(answer: str) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    for sentence in _SENTENCE_RE.split(answer.strip()):
        stripped_sentence = sentence.strip()
        if not stripped_sentence:
            continue
        text = strip_markers(stripped_sentence).strip(" \t\n-•")
        if len(text) < _MIN_CLAIM_CHARS:
            continue  # bullets, ellipses, noise — not claims
        claims.append(
            ExtractedClaim(
                text=text,
                raw=stripped_sentence,
                markers=extract_markers(stripped_sentence),
            )
        )
    return claims


def all_markers(answer: str) -> set[int]:
    """Every passage number referenced anywhere in the answer."""
    return {int(m) for m in MARKER_RE.findall(answer)}