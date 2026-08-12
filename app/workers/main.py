"""Worker entrypoint — PLACEHOLDER.

The real queue consumer (parsing → chunking → embedding → indexing with
retries and idempotency) lands in Phase 19. This stub keeps
`docker compose up` fully green from Phase 2 onward.
"""

from __future__ import annotations

import signal
import time
from types import FrameType

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger("worker")


def main() -> None:
    settings = get_settings()
    setup_logging(level=settings.app.log_level, json_output=settings.app.log_json)
    logger.info("worker_stub_started", note="real consumer implemented in Phase 19")

    stop = False

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop
        logger.info("worker_stub_signal_received", signal=signum)
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not stop:
        time.sleep(30)
        logger.info("worker_stub_heartbeat")

    logger.info("worker_stub_stopped")


if __name__ == "__main__":
    main()