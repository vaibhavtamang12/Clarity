"""Security event auditing (Phase 25).

Every security-relevant event lands in BOTH structured logs and a
Prometheus counter — auditable after the fact, observable in real time.

Log fields pass through the Phase 24 sanitize policy: event metadata is
operational (event type, labels, filenames) — never document content.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.observability.logging_policy import sanitize_fields
from app.observability.metrics import SECURITY_EVENTS_TOTAL

logger = get_logger("security")


def record_security_event(event_type: str, **fields: Any) -> None:
    """Record a security event: Prometheus counter + sanitized structured log."""
    SECURITY_EVENTS_TOTAL.labels(event_type=event_type).inc()
    logger.warning(
        "security_event",
        **sanitize_fields({"event_type": event_type, **fields}),
    )