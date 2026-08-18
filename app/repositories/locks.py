"""Per-document locks (repository layer — Phase 19).

Prevents concurrent processing of the same document (ingest racing reindex,
duplicate submissions slipping past idempotency under load).

Redis implementation: SET NX EX for acquisition; a Lua compare-and-delete
for release, so a worker whose lock expired can never delete another
worker's lock.
"""

from __future__ import annotations

import uuid

from redis.asyncio import Redis

from app.core.logging import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "lock:doc"
DEFAULT_TTL_SECONDS = 600

# Atomic release: delete only if we still hold the lock.
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class RedisDocumentLock:
    def __init__(
        self,
        client: Redis,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        prefix: str = KEY_PREFIX,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, document_id: str) -> str:
        return f"{self._prefix}:{document_id}"

    async def try_acquire(self, document_id: str) -> str | None:
        token = uuid.uuid4().hex
        acquired = await self._client.set(
            self._key(document_id), token, nx=True, ex=self._ttl
        )
        return token if acquired else None

    async def release(self, document_id: str, token: str) -> bool:
        result = await self._client.eval(self._RELEASE_SCRIPT, 1, self._key(document_id), token)
        released = bool(int(result))
        if not released:
            logger.warning(
                "document_lock_release_missed",
                document_id=document_id,
            )
        return released


class InMemoryDocumentLock:
    """Same CAS semantics, process-local (tests / Redis-down fallback)."""

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}

    async def try_acquire(self, document_id: str) -> str | None:
        if document_id in self._locks:
            return None
        token = uuid.uuid4().hex
        self._locks[document_id] = token
        return token

    async def release(self, document_id: str, token: str) -> bool:
        if self._locks.get(document_id) == token:
            del self._locks[document_id]
            return True
        return False