"""Sentence Transformers adapter (production embedder, ADR-002).

Properties that matter operationally:
- Lazy loading: the model downloads/loads on FIRST embed call, never at
  construction. Building this object is free; tests run without torch.
- Batching and normalization are delegated to SentenceTransformer.
- Query prefix from the registry (e.g. "query: " for E5 models).
- Missing dependency → EmbeddingUnavailableError, not an ImportError leak.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.core.exceptions import EmbeddingUnavailableError
from app.embeddings.registry import EmbeddingModelConfig

# Injectable for tests: (model_id, device) -> model-like object with .encode()
ModelLoader = Callable[[str, str], Any]


def _default_loader(model_id: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover — environment-dependent
        raise EmbeddingUnavailableError(
            "sentence-transformers is not installed; install the 'ml' extra"
        ) from exc
    return SentenceTransformer(model_id, device=device)


class SentenceTransformerModel:
    def __init__(
        self,
        config: EmbeddingModelConfig,
        model_key: str,
        device: str = "cpu",
        loader: ModelLoader | None = None,
    ) -> None:
        self.model_key = model_key
        self.model_id = config.model_id
        self.model_version = config.model_version
        self.dimension = config.dimension
        self.max_tokens = config.max_tokens
        self._config = config
        self._device = device
        self._loader = loader or _default_loader
        self._model: Any = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> Any:
        if self._model is None:
            try:
                model = self._loader(self.model_id, self._device)
            except EmbeddingUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 — normalize load failures
                raise EmbeddingUnavailableError(
                    f"Failed to load embedding model '{self.model_id}'"
                ) from exc
            # Respect the registry token budget if the model allows it.
            max_seq = getattr(model, "max_seq_length", None)
            if max_seq is not None:
                try:
                    model.max_seq_length = min(self.max_tokens, max_seq)
                except Exception:  # noqa: BLE001 — best effort only
                    pass
            self._model = model
        return self._model

    # ---------------------------------------------------------------- embedding
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return [[float(x) for x in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self._config.query_prefix}{text}"
        return self.embed_documents([prefixed])[0]