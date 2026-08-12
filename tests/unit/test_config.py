"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_without_env_file() -> None:
    s = Settings(_env_file=None)
    assert s.app.name == "rag-knowledge-platform"
    assert s.database.host == "localhost"
    assert s.retrieval.fusion_strategy == "rrf"
    assert not s.is_production


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE__HOST", "pg.internal")
    monkeypatch.setenv("DATABASE__PORT", "5433")
    monkeypatch.setenv("RETRIEVAL__FUSION_STRATEGY", "weighted")
    s = Settings(_env_file=None)
    assert s.database.host == "pg.internal"
    assert s.database.port == 5433
    assert s.retrieval.fusion_strategy == "weighted"


def test_database_url_construction() -> None:
    s = Settings(_env_file=None)
    url = s.database.async_url
    assert url.startswith("postgresql+asyncpg://")
    assert s.database.name in url


def test_password_is_url_encoded() -> None:
    s = Settings(_env_file=None, database={"password": "p@ss/word?"})
    assert "p%40ss%2Fword%3F" in s.database.async_url


def test_invalid_port_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database={"port": 99999})


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app={"log_level": "LOUD"})