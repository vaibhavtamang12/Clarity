"""RAG response cache (Phase 18, decision D-098).

Key structure (ARCHITECTURE §5.3):
    cache:rag:v1:{index_state}:{query_hash}

- index_state: current value of the collection's state counter. Any version
  change bumps the counter → keys change → old entries become unreachable
  and expire naturally. NO explicit invalidation, NO staleness race.
- query_hash: sha256(question + user_id + filters). Per-user keys mean
  tenant isolation is preserved by the cache itself (D-100).

Fail-open (D-101): any store failure degrades to "no cache" with a warning.
"""

from __future__ import annotations

import uuid

from pydantic import ValidationError

from app.core.config import CacheSettings
from app.core.logging import get_logger
from app.embeddings.caching import CacheStore, deserialize_vector  # noqa: F401 (store protocol)
from app.generation.domain import RAGResponse
from app.ingestion.domain import sha256_hex
from app.services.index_state import IndexStateNotifier

logger = get_logger(__name__)

KEY_PREFIX = "cache:rag:v1"


def is_cacheable(response: RAGResponse) -> bool:
    """Only clean, complete, standalone answers enter the cache.

    Degraded, uncertain, or policy-flagged responses must be regenerated —
    caching them would freeze a failure in place.
    """
    if response.mode != "generated":
        return False
    if response.degraded or response.insufficient_evidence:
        return False
    policy = response.grounding_policy.action if response.grounding_policy else None
    return policy in (None, "accept")


class ResponseCache:
    def __init__(
        self,
        store: CacheStore,
        index_state: IndexStateNotifier,
        settings: CacheSettings,
        collection_scope: str,
    ) -> None:
        self._store = store
        self._index_state = index_state
        self._settings = settings
        self._scope = collection_scope

    async def _key(self, user_id: uuid.UUID, question: str, filter_repr: str = "") -> str:
        state = await self._index_state.current(self._scope)
        payload = f"{question}|{user_id}|{filter_repr}"
        return f"{KEY_PREFIX}:{state}:{sha256_hex(payload)}"

    async def get(
        self, user_id: uuid.UUID, question: str, filter_repr: str = ""
    ) -> RAGResponse | None:
        if not self._settings.enabled:
            return None
        try:
            key = await self._key(user_id, question, filter_repr)
            raw_map = await self._store.get_many([key])
            raw = raw_map.get(key)
            if raw is None:
                return None
            return RAGResponse.model_validate_json(raw.decode("utf-8"))
        except (ValidationError, UnicodeDecodeError) as exc:
            logger.warning("response_cache_corrupt_entry", error=type(exc).__name__)
            return None
        except Exception as exc:  # noqa: BLE001 — fail-open (D-101)
            logger.warning("response_cache_get_failed", error=type(exc).__name__)
            return None

    async def put(
        self,
        user_id: uuid.UUID,
        question: str,
        response: RAGResponse,
        filter_repr: str = "",
    ) -> None:
        if not self._settings.enabled or not is_cacheable(response):
            return
        try:
            key = await self._key(user_id, question, filter_repr)
            await self._store.set(
                key,
                response.model_dump_json().encode("utf-8"),
                self._settings.rag_response_ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — fail-open (D-101)
            logger.warning("response_cache_put_failed", error=type(exc).__name__)