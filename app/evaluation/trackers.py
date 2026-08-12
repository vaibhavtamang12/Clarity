"""Experiment tracking abstraction (Rule 4).

MLflow is the production tracker; NoopTracker keeps experiments runnable
without an MLflow server (tests, CI, offline laptops). Selection is driven
by settings — never hard-coded (Rule 3).
"""

from __future__ import annotations

from typing import Protocol

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker(Protocol):
    def track_strategy(self, run_name: str, params: dict[str, str], metrics: dict[str, float]) -> None: ...


class NoopTracker:
    def track_strategy(self, run_name: str, params: dict[str, str], metrics: dict[str, float]) -> None:
        logger.debug("tracker_noop", run_name=run_name)


class MLflowTracker:
    """One MLflow run per strategy configuration."""

    def __init__(self, settings: Settings) -> None:
        import mlflow  # lazy: MLflow is heavy and optional at runtime

        self._mlflow = mlflow
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)

    def track_strategy(self, run_name: str, params: dict[str, str], metrics: dict[str, float]) -> None:
        with self._mlflow.start_run(run_name=run_name):
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
        logger.info("mlflow_run_logged", run_name=run_name)


def build_tracker(settings: Settings) -> ExperimentTracker:
    if not settings.mlflow.enabled:
        return NoopTracker()
    try:
        return MLflowTracker(settings)
    except Exception as exc:  # noqa: BLE001 — tracking must never break experiments
        logger.warning("mlflow_unavailable_falling_back_to_noop", error=type(exc).__name__)
        return NoopTracker()