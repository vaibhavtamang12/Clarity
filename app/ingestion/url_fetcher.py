"""URL fetching with hop-by-hop SSRF validation.

Redirects are followed manually (max 5 hops) and EVERY hop is re-validated
before the request is made — an open redirect on a trusted host cannot be
used to reach internal addresses (decision D-028).
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import IngestionSettings
from app.core.exceptions import IngestionValidationError
from app.ingestion.validation import validate_url

MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 20.0


@dataclass
class FetchedContent:
    content: bytes
    content_type: str
    final_url: str


async def fetch_url(
    raw_url: str, settings: IngestionSettings, max_bytes: int | None = None
) -> FetchedContent:
    max_bytes = max_bytes or settings.max_file_size_mb * 1024 * 1024
    url = validate_url(raw_url, settings)

    async with httpx.AsyncClient(
        follow_redirects=False, timeout=FETCH_TIMEOUT_SECONDS
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            response = await client.get(url)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise IngestionValidationError("Redirect with no Location header")
                url = validate_url(urljoin(str(response.url), location), settings)
                continue
            response.raise_for_status()
            content = response.content
            if len(content) > max_bytes:
                raise IngestionValidationError("Fetched content exceeds the size limit")
            return FetchedContent(
                content=content,
                content_type=response.headers.get("content-type", ""),
                final_url=str(response.url),
            )
    raise IngestionValidationError(f"Too many redirects (>{MAX_REDIRECTS})")