"""Worker entrypoint — production implementation (Phase 19).

Runs the ingestion worker process:
- Redis stream consumer (competing consumers via consumer groups)
- PostgreSQL sweep as the safety net / Redis-down mode
- per-document locks, taxonomy-driven retries, dead-letter queue

Usage:
    python -m app.workers.main
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket

from app.container import build_platform
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.repositories.database import Database
from app.repositories.redis_client import build_redis_client
from app.workers.worker import IngestionWorker

logger = get_logger("workers.main")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    database = Database(settings.database)
    await database.initialize()

    # Redis is optional at runtime: without it the worker degrades to pure
    # DB polling (D-101 lineage). Correctness is unaffected; wake-up latency
    # rises to poll_interval_seconds.
    redis_client = build_redis_client(settings.redis)
    redis_for_platform = None
    try:
        if await redis_client.ping():
            redis_for_platform = redis_client
    except Exception:  # noqa: BLE001
        logger.warning("worker_redis_unavailable_polling_mode")

    platform = build_platform(settings, database, redis_client=redis_for_platform)

    consumer_name = f"{socket.gethostname()}-{os.getpid()}"
    worker = IngestionWorker(
        runner=platform.job_runner,
        queue=platform.job_queue,
        settings=settings.worker,
        consumer_name=consumer_name,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, worker.request_stop)
        except NotImplementedError:  # pragma: no cover — non-unix platforms
            pass

    try:
        await worker.run()
    finally:
        if redis_for_platform is not None:
            await redis_client.close()
        await database.dispose()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())