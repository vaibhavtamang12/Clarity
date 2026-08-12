"""Key handling utilities.

API keys are stored as SHA-256 hashes only — the plaintext value is shown
to the user exactly once at creation time and can never be recovered.
"""

from __future__ import annotations

import hashlib
import secrets

API_KEY_PREFIX = "rk_"


def generate_api_key() -> str:
    """Generate a high-entropy API key (256 bits of randomness)."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()