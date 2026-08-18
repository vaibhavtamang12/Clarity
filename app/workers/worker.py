"""Ingestion worker loop (Phase 19).

Execution model — PostgreSQL is truth, the queue is a trigger:
1. Stream message arrives → runner.run_specific(job_id)
   (idempotent: redelivered or already-processed jobs are skipped)
2. Periodic DB sweep via runner.run_next() catches everything the stream
   missed: Redis outage, publish failure, crash between process and ack
3. Messages are acked AFTER the DB state is resolved → at-least-once;
   idempotent execution makes duplicates safe
4. Startup recovers this consumer's pending (unacked) messages

Logs carry job ids, statuses, and error classes only — never document
content (Rule 10).
"""

from __future__ import annotations

import time

from app.core.config import WorkerSettings
from app.core.logging import get_logger
from app.services.ingestion_job_runner import IngestionJobRunner
from app.workers.protocols import JobQueue, QueueMessage

logger = get_logger(__name__)


class IngestionWorker:
    def __init__(
        self,
        runner: IngestionJobRunner,
        queue: JobQueue,
        settings: WorkerSettings,
        consumer_name: str = "worker-1",
    ) -> None:
        self._runner = runner
        self._queue = queue
        self._settings = settings
        self._consumer_name = consumer_name
        self._stopping = False

    # ---------------------------------------------------------------- control
    def request_stop(self) -> None:
        logger.info("worker_stop_requested", consumer=self._consumer_name)
        self._stopping = True

    # ------------------------------------------------------------------- main
    async def run(self) -> None:
        await self._queue.ensure_group()
        logger.info(
            "worker_started",
            consumer=self._consumer_name,
            poll_interval_seconds=self._settings.poll_interval_seconds,
        )

        # Crash recovery: process anything we received but never acked.
        for message in await self._queue.recover_pending():
            if self._stopping:
                break
            await self._handle(message)

        last_sweep = 0.0
        while not self._stopping:
            message = await self._queue.receive(block_ms=self._settings.block_ms)
            if message is not None:
                await self._handle(message)
                continue

            # Idle: run the DB sweep on schedule (safety net + Redis-down mode).
            now = time.monotonic()
            if now - last_sweep >= self._settings.poll_interval_seconds:
                last_sweep = now
                try:
                    await self._runner.run_next()
                except Exception as exc:  # noqa: BLE001 — sweep failures must not kill the loop
                    logger.warning("worker_sweep_failed", error=type(exc).__name__)

        logger.info("worker_stopped", consumer=self._consumer_name)

    # ------------------------------------------------------------------ single
    async def run_once(self) -> bool:
        """Process one message (or one sweep job when the queue is empty).
        Exposed for tests and scripts."""
        message = await self._queue.receive(block_ms=0)
        if message is not None:
            await self._handle(message)
            return True
        return await self._runner.run_next()

    # ---------------------------------------------------------------- internals
    async def _handle(self, message: QueueMessage) -> None:
        try:
            processed = await self._runner.run_specific(message.job_id)
            logger.info(
                "worker_message_processed",
                job_id=str(message.job_id),
                processed=processed,
            )
        except Exception as exc:  # noqa: BLE001 — DB state already resolved inside runner
            logger.warning(
                "worker_message_error",
                job_id=str(message.job_id),
                error=type(exc).__name__,
            )
        finally:
            # Ack after DB resolution. If ack itself fails, redelivery is
            # harmless: run_specific skips non-QUEUED jobs (idempotency).
            try:
                await self._queue.ack(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("worker_ack_failed", error=type(exc).__name__)