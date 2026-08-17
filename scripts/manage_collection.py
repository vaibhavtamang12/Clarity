#!/usr/bin/env python3
"""Operational CLI for Qdrant collections.

Usage:
    python scripts/manage_collection.py info                 # exists? count?
    python scripts/manage_collection.py create               # create for default model
    python scripts/manage_collection.py delete --yes         # drop collection

Collection naming follows ADR-002: {prefix}__chunks__{model_key}.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.embeddings.naming import collection_name
from app.embeddings.registry import load_embedding_registry
from app.repositories.vector.qdrant_client import build_qdrant_client
from app.repositories.vector.qdrant_repository import QdrantVectorRepository

REPO_ROOT = Path(__file__).resolve().parents[1]


async def run(action: str, model_key: str, assume_yes: bool) -> None:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_json)

    client = build_qdrant_client(settings.qdrant)
    try:
        name = collection_name(settings.qdrant.collection_prefix, model_key)
        repo = QdrantVectorRepository(client, name)

        if action == "info":
            exists = await repo.collection_exists()
            print(f"collection: {name}")
            print(f"exists: {exists}")
            if exists:
                print(f"points: {await repo.count()}")
        elif action == "create":
            registry = load_embedding_registry(REPO_ROOT / "configs" / "embeddings.yaml")
            dimension = registry.models[model_key].dimension
            await repo.ensure_collection(dimension)
            print(f"collection ready: {name} (dimension={dimension})")
        elif action == "delete":
            if not assume_yes:
                raise SystemExit("refusing to delete without --yes")
            await repo.delete_collection()
            print(f"deleted: {name}")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["info", "create", "delete"])
    parser.add_argument("--model-key", default=None, help="defaults to EMBEDDING__DEFAULT_MODEL")
    parser.add_argument("--yes", action="store_true", help="required for delete")
    args = parser.parse_args()

    settings = get_settings()
    asyncio.run(run(args.action, args.model_key or settings.embedding.default_model, args.yes))