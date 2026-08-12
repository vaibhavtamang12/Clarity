"""Health endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_liveness(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_v1_shape(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "rag-knowledge-platform"
    assert body["environment"] == "test"
    assert set(body["checks"]) == {"postgres", "redis", "qdrant"}
    # PostgreSQL is probed for real now: healthy if the local DB is up,
    # unavailable otherwise. Either is a valid, honest outcome in tests.
    assert body["checks"]["postgres"]["status"] in {"healthy", "unavailable"}
    assert body["checks"]["redis"]["status"] == "not_configured"


def test_request_id_generated(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.headers.get("X-Request-ID")


def test_request_id_propagated(client: TestClient) -> None:
    resp = client.get("/healthz", headers={"X-Request-ID": "test-rid-123"})
    assert resp.headers["X-Request-ID"] == "test-rid-123"