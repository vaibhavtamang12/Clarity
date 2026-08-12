"""Safe YAML loading shared by all config registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.core.exceptions import ConfigurationError


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a mapping: {p}")
    return data