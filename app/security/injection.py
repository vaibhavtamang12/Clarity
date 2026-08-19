"""Prompt-injection defense (Phase 25, decisions D-137/D-138).

Doctrine: retrieved document content is UNTRUSTED DATA, never instructions.

Two layers, in order of importance:

1. STRUCTURAL ISOLATION (primary defense):
   Passage content is XML-escaped before entering the prompt, so a document
   can never close a <passage> tag, open new tags, or impersonate prompt
   structure. This works against ANY injection payload, known or unknown,
   because it removes the *mechanism* of escape rather than matching words.

2. PATTERN DETECTION (defense-in-depth, alerting only):
   Known injection signatures are scanned and flagged as security events.
   Detection NEVER blocks ingestion — blocking on content patterns is a
   denial-of-service vector (a competitor could poison your corpus with
   trigger words). Flag + alert + let the structural layer defend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I), "ignore_instructions"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.I), "disregard_previous"),
    (re.compile(r"you\s+are\s+now\b", re.I), "role_hijack"),
    (re.compile(r"(new|updated)\s+system\s+prompt", re.I), "system_prompt_override"),
    (re.compile(r"reveal\s+(your\s+)?(system|initial|secret)\s+prompt", re.I), "prompt_exfiltration"),
    (re.compile(r"repeat\s+(the\s+)?(system|above)\s+(prompt|instructions)", re.I), "prompt_exfiltration"),
    (re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M), "role_marker"),
    (re.compile(r"do\s+not\s+follow\s+(the\s+)?(above|previous|earlier)", re.I), "instruction_override"),
    (re.compile(r"\b(jailbreak|DAN\s+mode|developer\s+mode)\b", re.I), "jailbreak_keyword"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(?!a\s+(customer|user))", re.I), "persona_hijack"),
)


@dataclass(frozen=True)
class InjectionScanResult:
    suspicious: bool
    patterns_matched: tuple[str, ...]

    @property
    def score(self) -> int:
        return len(self.patterns_matched)


def scan_for_injection(text: str) -> InjectionScanResult:
    """Scan text for known injection signatures (alerting, never blocking)."""
    matched: list[str] = []
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(label)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = tuple(label for label in matched if not (label in seen or seen.add(label)))
    return InjectionScanResult(suspicious=bool(unique), patterns_matched=unique)


def escape_passage_content(text: str) -> str:
    """Escape content so it can never break out of passage tags.

    This is the PRIMARY injection defense: it removes the escape mechanism
    itself, independent of what the payload says.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attribute_value(value: str) -> str:
    """Escape values used inside XML tag attributes."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )