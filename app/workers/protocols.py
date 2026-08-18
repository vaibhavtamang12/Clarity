"""Worker-layer protocols (Phase 19).

The Phase 2 import-linter contract forbids app.workers from importing the
Redis client — only the repository layer may. So the worker programs
against these protocols, and the Redis implementations live in
app/repositories/ (job_queue.py, locks.py). Swapping queue technology is a
repository-layer change; the worker never notices (Rule 4).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    job_id: uuid.UUID


@runtime_checkable
class JobQueue(Protocol):
    """At-least-once job trigger channel. PostgreSQL remains the system of
    record; the queue only wakes workers faster than the polling sweep."""

    async def ensure_group(self) -> None: ...

    async def publish(self, job_id: uuid.UUID) -> None: ...

    async def receive(self, block_ms: int = 1000) -> QueueMessage | None: ...

    async def ack(self, message: QueueMessage) -> None: ...

    async def recover_pending(self) -> list[QueueMessage]: ...


@runtime_checkable
class DeadLetterSink(Protocol):
    async def send_dead_letter(
        self,
        job_id: uuid.UUID,
        error_type: str,
        error_message: str,
        reason: str,
    ) -> None: ...


@runtime_checkable
class DocumentLockManager(Protocol):
    """Per-document mutual exclusion (ingest vs reindex races)."""

    async def try_acquire(self, document_id: str) -> str | None:
        """Returns a lock token, or None when the lock is held elsewhere."""
        ...

    async def release(self, document_id: str, token: str) -> bool:
        """Atomic compare-and-delete: only the holder may release."""
        ...