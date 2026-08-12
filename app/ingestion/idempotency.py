"""Idempotency key computation for ingestion jobs.

key = sha256(content_hash + parser + chunking config + embedding model).
Same content with the same pipeline config → same key → the existing job is
returned instead of duplicate work. Change any input → new job (FR-1.5).
"""

from __future__ import annotations

import json

from app.ingestion.domain import sha256_hex


def compute_idempotency_key(
    *,
    content_hash: str,
    parser: str,
    chunking_strategy: str,
    chunking_config: dict[str, object],
    embedding_model: str,
) -> str:
    payload = json.dumps(
        {
            "content_hash": content_hash,
            "parser": parser,
            "chunking_strategy": chunking_strategy,
            "chunking_config": chunking_config,
            "embedding_model": embedding_model,
        },
        sort_keys=True,
    )
    return sha256_hex(payload)