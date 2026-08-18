"""Chunk-level version diffing (decision D-080).

Compares two versions by chunk content hashes using MULTISET semantics —
order-independent and duplicate-aware, so a chunk that appears twice in v1
and once in v2 correctly reports one removal.

This powers:
- compare_versions (what changed between releases)
- future selective reprocessing (only changed chunks need new treatment)
- audit trails for compliance-sensitive documents
"""

from __future__ import annotations

from collections import Counter, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VersionDiff:
    added: int
    removed: int
    unchanged: int
    added_hashes: tuple[str, ...]
    removed_hashes: tuple[str, ...]

    @property
    def total_changes(self) -> int:
        return self.added + self.removed


def diff_chunks(old_hashes: Sequence[str], new_hashes: Sequence[str]) -> VersionDiff:
    """Multiset diff: what v2 added vs v1, what v1 lost, what survived."""
    old_counter = Counter(h for h in old_hashes if h)
    new_counter = Counter(h for h in new_hashes if h)

    added_counter = new_counter - old_counter   # in new, beyond what old had
    removed_counter = old_counter - new_counter  # in old, beyond what new keeps

    unchanged = sum((old_counter & new_counter).values())

    return VersionDiff(
        added=sum(added_counter.values()),
        removed=sum(removed_counter.values()),
        unchanged=unchanged,
        added_hashes=tuple(sorted(added_counter.elements())),
        removed_hashes=tuple(sorted(removed_counter.elements())),
    )