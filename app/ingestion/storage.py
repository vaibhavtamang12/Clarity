"""File storage abstraction (decision D-027).

Uploaded/fetched bytes are stored once at submission time; workers replay
the stored bytes on every attempt. This makes retries deterministic and
idempotent — a retry never re-fetches a URL or re-reads a request body.
LocalFileStore writes atomically (tmp file + rename) so a crash can never
leave a half-written file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.exceptions import IngestionError


class FileStore(Protocol):
    def save(self, key: str, data: bytes) -> None: ...
    def load(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalFileStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = Path(key).name  # defense in depth: no path traversal
        return self.root / safe

    def save(self, key: str, data: bytes) -> None:
        target = self._path(key)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    def load(self, key: str) -> bytes:
        target = self._path(key)
        if not target.exists():
            raise IngestionError(f"Stored file missing for key '{key}'")
        return target.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()