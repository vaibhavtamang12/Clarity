"""Error envelope tests — every failure mode returns one stable shape."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import DocumentNotFoundError


def test_domain_error_envelope(app: FastAPI, client: TestClient) -> None:
    @app.get("/_test/domain-error")
    async def raise_domain() -> None:
        raise DocumentNotFoundError("Document 123 not found")

    resp = client.get("/_test/domain-error")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "DOCUMENT_NOT_FOUND"
    assert error["message"] == "Document 123 not found"
    assert error["request_id"]


def test_validation_error_envelope(app: FastAPI, client: TestClient) -> None:
    @app.get("/_test/validation")
    async def needs_int(q: int) -> dict[str, int]:
        return {"q": q}

    resp = client.get("/_test/validation", params={"q": "not-an-int"})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["fields"]


def test_unknown_route_uses_envelope(client: TestClient) -> None:
    resp = client.get("/api/v1/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"