"""Shared test fixtures.

Every test runs against an isolated Settings instance (cache cleared before
and after) with forced test-environment values, so no test can leak state
into another via the cached settings singleton.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("APP__ENVIRONMENT", "test")
    monkeypatch.setenv("APP__LOG_JSON", "false")
    monkeypatch.setenv("APP__LOG_LEVEL", "WARNING")

    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def app() -> FastAPI:
    from app.main import create_app

    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client