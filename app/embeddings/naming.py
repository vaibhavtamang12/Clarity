"""Qdrant collection naming — one collection per embedding model (ADR-002).

Changing the embedding model creates a NEW collection; the old one keeps
serving traffic while the new one is backfilled (blue/green re-index).
No in-place mutation, no mixed-model vectors.
"""

from __future__ import annotations


def collection_name(prefix: str, model_key: str) -> str:
    if prefix:
        return f"{prefix}__chunks__{model_key}"
    return f"chunks__{model_key}"