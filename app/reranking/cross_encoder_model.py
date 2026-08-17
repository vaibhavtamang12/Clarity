"""Cross-encoder reranker adapter (production, ADR-004).

A cross-encoder reads (query, chunk) jointly — materially more precise than
bi-encoder similarity, at ~100x the per-pair cost. That asymmetry is exactly
why the pipeline retrieves wide (top-50) and reranks narrow (top-10).

Properties:
- Lazy loading: no weights touched until the first rerank call.
- Batched scoring with configurable batch size.
- Injectable loader → wrapper logic testable without torch.
- Load/score failures normalize to RerankerUnavailableError.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.core.exceptions import RerankerUnavailableError
from app.reranking.registry import RerankerModelConfig
from app.retrieval.base import RetrievedChunk

ModelLoader = Callable[[str, str], Any]  # (model_id, device) -> model with .predict()


def _default_loader(model_id: str, device: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover — environment-dependent
        raise RerankerUnavailableError(
            "sentence-transformers is not installed; install the 'ml' extra"
        ) from exc
    return CrossEncoder(model_id, device=device)


class CrossEncoderReranker:
    def __init__(
        self,
        config: RerankerModelConfig,
        model_key: str,
        device: str = "cpu",
        loader: ModelLoader | None = None,
    ) -> None:
        self.model_key = model_key
        self.model_id = config.model_id
        self._device = device
        self._batch_size = config.batch_size
        self._loader = loader or _default_loader
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                self._model = self._loader(self.model_id, self._device)
            except RerankerUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 — normalize load failures
                raise RerankerUnavailableError(
                    f"Failed to load reranker model '{self.model_id}'"
                ) from exc
        return self._model

    def rerank(self, query: str, items: Sequence[RetrievedChunk]) -> list[float]:
        if not items:
            return []
        model = self._load()
        pairs = [(query, item.content) for item in items]
        try:
            raw_scores = model.predict(
                pairs, batch_size=self._batch_size, show_progress_bar=False
            )
        except Exception as exc:  # noqa: BLE001 — normalize scoring failures
            raise RerankerUnavailableError(
                f"Reranker scoring failed: {type(exc).__name__}"
            ) from exc
        return [float(s) for s in raw_scores]