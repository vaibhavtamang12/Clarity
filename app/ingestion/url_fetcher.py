"""URL fetching with hop-by-hop SSRF validation and streaming size caps.

Redirects are followed manually (max 5 hops) and EVERY hop is re-validated
before the request is made — an open redirect on a trusted host cannot reach
internal addresses.

Phase 25 hardening (D-140): response bodies are streamed with an incremental
size cap. A malicious URL can no longer exhaust memory by serving a huge
body — the fetch aborts the moment the cap is crossed.

Residual risk (D-141): DNS rebinding between validation and connection is
not fully mitigated without IP-pinned transport; hop revalidation covers
redirect-based SSRF, which is the dominant attack path.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import IngestionSettings
from app.core.exceptions import IngestionValidationError
from app.ingestion.validation import validate_url
from app.security.audit import record_security_event

MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 20.0
CHUNK_SIZE = 65536


@dataclass
class FetchedContent:
    content: bytes
    content_type: str
    final_url: str


async def fetch_url(
    raw_url: str, settings: IngestionSettings, max_bytes: int | None = None
) -> FetchedContent:
    max_bytes = max_bytes or settings.max_file_size_mb * 1024 * 1024
    try:
        url = validate_url(raw_url, settings)
    except IngestionValidationError:
        record_security_event("ssrf_blocked", scheme="validate", source=raw_url[:200])
        raise

    async with httpx.AsyncClient(
        follow_redirects=False, timeout=FETCH_TIMEOUT_SECONDS
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise IngestionValidationError("Redirect with no Location header")
                    next_url = urljoin(str(response.url), location)
                    try:
                        url = validate_url(next_url, settings)
                    except IngestionValidationError:
                        record_security_event(
                            "ssrf_blocked", scheme="redirect", source=next_url[:200]
                        )
                        raise
                    continue

                response.raise_for_status()

                # Streaming size cap: abort before memory exhaustion (D-140).
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise IngestionValidationError(
                            "Fetched content exceeds the size limit"
                        )
                    chunks.append(chunk)

                return FetchedContent(
                    content=b"".join(chunks),
                    content_type=response.headers.get("content-type", ""),
                    final_url=str(response.url),
                )
    raise IngestionValidationError(f"Too many redirects (>{MAX_REDIRECTS})")