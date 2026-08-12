"""Unit tests for upload + URL validation (including SSRF blocking)."""

from __future__ import annotations

import socket

import pytest

from app.core.config import IngestionSettings
from app.core.exceptions import IngestionValidationError
from app.ingestion.validation import validate_upload, validate_url
from tests.fixtures.builders import build_pdf


@pytest.fixture()
def settings() -> IngestionSettings:
    return IngestionSettings()


def test_valid_pdf_passes(settings: IngestionSettings) -> None:
    content = build_pdf([[("Hello", 12, False)]])
    assert validate_upload("doc.pdf", len(content), content, settings).value == "pdf"


def test_extension_mismatch_rejected(settings: IngestionSettings) -> None:
    with pytest.raises(IngestionValidationError):
        validate_upload("fake.pdf", 8, b"notapdf!", settings)


def test_oversized_file_rejected(settings: IngestionSettings) -> None:
    settings.max_file_size_mb = 1
    with pytest.raises(IngestionValidationError):
        validate_upload("big.txt", 2 * 1024 * 1024, b"x", settings)


def test_unknown_extension_rejected(settings: IngestionSettings) -> None:
    with pytest.raises(IngestionValidationError):
        validate_upload("malware.exe", 10, b"MZ........", settings)


def _fake_dns(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (ip, 0))]
    )


def test_url_blocks_loopback(monkeypatch: pytest.MonkeyPatch, settings: IngestionSettings) -> None:
    _fake_dns(monkeypatch, "127.0.0.1")
    with pytest.raises(IngestionValidationError):
        validate_url("http://internal.example.com/x", settings)


def test_url_blocks_cloud_metadata(monkeypatch: pytest.MonkeyPatch, settings: IngestionSettings) -> None:
    _fake_dns(monkeypatch, "169.254.169.254")
    with pytest.raises(IngestionValidationError):
        validate_url("http://spoofed.example.com/latest/meta-data", settings)


def test_url_blocks_private_ranges(monkeypatch: pytest.MonkeyPatch, settings: IngestionSettings) -> None:
    _fake_dns(monkeypatch, "10.1.2.3")
    with pytest.raises(IngestionValidationError):
        validate_url("https://wiki.internal/page", settings)


def test_url_blocks_bad_scheme(settings: IngestionSettings) -> None:
    with pytest.raises(IngestionValidationError):
        validate_url("ftp://example.com/file", settings)


def test_url_blocks_embedded_credentials(settings: IngestionSettings) -> None:
    with pytest.raises(IngestionValidationError):
        validate_url("https://user:pass@example.com/", settings)


def test_url_allows_public_address(monkeypatch: pytest.MonkeyPatch, settings: IngestionSettings) -> None:
    _fake_dns(monkeypatch, "93.184.216.34")
    assert validate_url("https://example.com/page", settings) == "https://example.com/page"