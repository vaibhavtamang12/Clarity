"""Input validation: file uploads and URLs (Phase 4; Phase 25 adds binary safety).

File checks: extension allow-list, size cap, magic-byte sniffing, then
binary safety validation (page caps, zip bomb/slip guards) — all BEFORE any
parser executes.

URL checks (SSRF posture): scheme allow-list, no embedded credentials, DNS
resolution, rejection of private/loopback/link-local/reserved addresses —
including the cloud metadata endpoint 169.254.169.254. Redirects are
validated hop-by-hop in url_fetcher.py.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import IngestionSettings
from app.core.exceptions import IngestionValidationError
from app.models.enums import DocumentSourceType
from app.security.file_safety import validate_binary_safety

_MAGIC_TO_TYPE: dict[bytes, DocumentSourceType] = {
    b"%PDF": DocumentSourceType.PDF,
    b"PK\x03\x04": DocumentSourceType.DOCX,
}

_EXTENSION_TO_TYPE: dict[str, DocumentSourceType] = {
    ".pdf": DocumentSourceType.PDF,
    ".docx": DocumentSourceType.DOCX,
    ".md": DocumentSourceType.MARKDOWN,
    ".markdown": DocumentSourceType.MARKDOWN,
    ".txt": DocumentSourceType.TXT,
    ".html": DocumentSourceType.HTML,
    ".htm": DocumentSourceType.HTML,
}

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def detect_source_type(filename: str) -> DocumentSourceType:
    ext = Path(filename).suffix.lower()
    source_type = _EXTENSION_TO_TYPE.get(ext)
    if source_type is None:
        raise IngestionValidationError(
            f"Unsupported file extension '{ext or '<none>'}'",
            details={"allowed": sorted(set(_EXTENSION_TO_TYPE))},
        )
    return source_type


def validate_upload(
    filename: str, size_bytes: int, content: bytes, settings: IngestionSettings
) -> DocumentSourceType:
    source_type = detect_source_type(filename)

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionValidationError(
            f"File exceeds the {settings.max_file_size_mb} MB limit",
            details={"max_bytes": max_bytes, "received_bytes": size_bytes},
        )
    if size_bytes == 0:
        raise IngestionValidationError("File is empty")

    sniffed = _sniff_magic(content)
    if source_type in (DocumentSourceType.PDF, DocumentSourceType.DOCX):
        if sniffed != source_type:
            raise IngestionValidationError(
                "File content does not match its extension",
                details={"extension_type": source_type.value},
            )
        # Phase 25: binary safety BEFORE any parser runs (D-139).
        validate_binary_safety(content, source_type, settings)

    return source_type


def _sniff_magic(content: bytes) -> DocumentSourceType | None:
    for magic, source_type in _MAGIC_TO_TYPE.items():
        if content.startswith(magic):
            return source_type
    return None


def validate_url(raw_url: str, settings: IngestionSettings) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in settings.allowed_url_schemes:
        raise IngestionValidationError(
            "URL scheme not allowed", details={"allowed": settings.allowed_url_schemes}
        )
    if parsed.username or parsed.password:
        raise IngestionValidationError("URLs with embedded credentials are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise IngestionValidationError("URL has no hostname")

    try:
        addr_info = socket.getaddrinfo(
            hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise IngestionValidationError("URL hostname could not be resolved") from exc

    for _family, _type, _proto, _canon, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast
        ):
            raise IngestionValidationError("URL resolves to a blocked network address")
        if any(ip in network for network in _BLOCKED_NETWORKS):
            raise IngestionValidationError("URL resolves to a blocked network address")
    return raw_url