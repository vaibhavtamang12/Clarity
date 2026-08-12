"""Unit tests for API key generation/hashing."""

from __future__ import annotations

from app.utils.security import API_KEY_PREFIX, generate_api_key, hash_api_key


def test_generated_key_shape() -> None:
    key = generate_api_key()
    assert key.startswith(API_KEY_PREFIX)
    assert len(key) >= 40  # 256 bits of entropy, base64url-encoded


def test_generated_keys_are_unique() -> None:
    assert generate_api_key() != generate_api_key()


def test_hash_is_deterministic_and_not_reversible_to_plaintext() -> None:
    key = generate_api_key()
    digest = hash_api_key(key)
    assert digest == hash_api_key(key)
    assert len(digest) == 64  # SHA-256 hex
    assert key not in digest