"""Structured logging built on structlog.

JSON output outside dev, pretty console in dev. Every log line can carry
``request_id`` via contextvars (bound by the request middleware).

Rule 10: logs contain concise operational metadata — never document content
unless explicitly enabled in a later phase.
"""

from __future__ import annotations

import logging
import sys

import structlog

_NOISY_LOGGERS = ("httpx", "httpcore", "asyncio", "urllib3", "watchfiles")


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    shared_processors: list[object] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: object = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_request_context(request_id: str, **extra: object) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()