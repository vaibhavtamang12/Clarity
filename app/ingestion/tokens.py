"""Token counting abstraction.

Chunk budgets are expressed in tokens. We deliberately avoid coupling to a
specific model tokenizer at this stage: the heuristic counter (≈4 chars/token)
is deterministic, dependency-free, and consistent across experiments — which
is what Phase 5 comparisons need. Swap in an exact tokenizer (tiktoken or a
model tokenizer) by implementing TokenCounter; nothing else changes (Rule 4).
"""

from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """≈4 characters per token. Deterministic and comparable across runs."""

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, math.ceil(len(stripped) / self.CHARS_PER_TOKEN))