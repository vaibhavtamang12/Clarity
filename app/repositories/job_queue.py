"""Job queue implementations (repository layer — the only layer allowed to
import the Redis client, per the Phase 2 import-linter contract).

Redis Streams with consumer groups provide:
- competing consumers (horizontal worker scaling)
- at-least-once delivery (ack after DB state is resolved)
- pending recovery after crashes (recover_pending on startup)
- a dead-letter stream for jobs that can never succeed

PostgreSQL remains the system of record: the queue is a trigger, the DB is
the truth. A lost message costs up to one poll_interval of latency — the DB
sweep catches everything.
"""

from __future__ import annotations

import uuid
from collections import deque

from redis.asyncio import Redis

from app.core.logging import get_logger
from app.workers.protocols import QueueMessage

logger = get_logger(__name__)

DEFAULT_STREAM_KEY = "queue:ingest"
DEFAULT_DLQ_KEY = "queue:ingest:dlq"
DEFAULT_GROUP = "ingestion-workers"


class RedisStreamJobQueue:
    """Redis Streams implementation of JobQueue + DeadLetterSink."""

    def __init__(
        self,
        client: Redis,
        stream_key: str = DEFAULT_STREAM_KEY,
        group_name: str = DEFAULT_GROUP,
        consumer_name: str = "worker-1",
        dlq_key: str = DEFAULT_DLQ_KEY,
    ) -> None:
        self._client = client
        self._stream = stream_key
        self._group = group_name
        self._consumer = consumer_name
        self._dlq = dlq_key

    # ------------------------------------------------------------------ setup
    async def ensure_group(self) -> None:
        try:
            await self._client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
        except Exception as exc:  # BUSYGROUP = group already exists
            if "BUSYGROUP" not in str(exc):
                raise

    # ---------------------------------------------------------------- publish
    async def publish(self, job_id: uuid.UUID) -> None:
        await self._client.xadd(self._stream, {"job_id": str(job_id)})

    # ---------------------------------------------------------------- receive
    async def receive(self, block_ms: int = 1000) -> QueueMessage | None:
        response = await self._client.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams={self._stream: ">"},
            count=1,
            block=block_ms,
        )
        if not response:
            return None
        _stream_name, messages = response[0]
        message_id, fields = messages[0]
        raw_job_id = fields.get(b"job_id") or fields.get("job_id")
        if raw_job_id is None:
            logger.warning("queue_message_missing_job_id", message_id=str(message_id))
            await self.ack(QueueMessage(message_id=str(message_id), job_id=uuid.uuid4()))
            return None
        if isinstance(raw_job_id, bytes):
            raw_job_id = raw_job_id.decode("utf-8")
        return QueueMessage(message_id=str(message_id), job_id=uuid.UUID(raw_job_id))

    async def ack(self, message: QueueMessage) -> None:
        await self._client.xack(self._stream, self._group, message.message_id)

    async def recover_pending(self) -> list[QueueMessage]:
        """Messages this consumer received but never acked (crash recovery).

        Reading with id '0' returns this consumer's pending entries instead
        of new ones — at-least-once delivery made explicit.
        """
        response = await self._client.xreadgroup(
            groupname=self._group,
            consumername=self._consumer,
            streams={self._stream: "0"},
            count=100,
        )
        pending: list[QueueMessage] = []
        if not response:
            return pending
        _stream_name, messages = response[0]
        for message_id, fields in messages:
            raw_job_id = fields.get(b"job_id") or fields.get("job_id")
            if raw_job_id is None:
                continue
            if isinstance(raw_job_id, bytes):
                raw_job_id = raw_job_id.decode("utf-8")
            pending.append(
                QueueMessage(message_id=str(message_id), job_id=uuid.UUID(raw_job_id))
            )
        return pending

    # -------------------------------------------------------------------- dlq
    async def send_dead_letter(
        self,
        job_id: uuid.UUID,
        error_type: str,
        error_message: str,
        reason: str,
    ) -> None:
        await self._client.xadd(
            self._dlq,
            {
                "job_id": str(job_id),
                "error_type": error_type[:200],
                "error_message": error_message[:2000],
                "reason": reason,
            },
        )
        logger.warning(
            "job_sent_to_dlq", job_id=str(job_id), reason=reason, error_type=error_type
        )


class InMemoryJobQueue:
    """In-process queue with identical semantics: tests, single-process dev,
    and the Redis-down fallback (D-101 lineage)."""

    def __init__(self) -> None:
        self._pending: deque[QueueMessage] = deque()
        self._unacked: dict[str, QueueMessage] = {}
        self.dead_letters: list[dict[str, str]] = []
        self._counter = 0

    async def ensure_group(self) -> None:
        return None

    async def publish(self, job_id: uuid.UUID) -> None:
        self._counter += 1
        self._pending.append(QueueMessage(message_id=f"m-{self._counter}", job_id=job_id))

    async def receive(self, block_ms: int = 1000) -> QueueMessage | None:
        if not self._pending:
            return None
        message = self._pending.popleft()
        self._unacked[message.message_id] = message
        return message

    async def ack(self, message: QueueMessage) -> None:
        self._unacked.pop(message.message_id, None)

    async def recover_pending(self) -> list[QueueMessage]:
        return list(self._unacked.values())

    async def send_dead_letter(
        self,
        job_id: uuid.UUID,
        error_type: str,
        error_message: str,
        reason: str,
    ) -> None:
        self.dead_letters.append(
            {
                "job_id": str(job_id),
                "error_type": error_type,
                "error_message": error_message,
                "reason": reason,
            }
        )