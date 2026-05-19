"""MLflow tracking abstraction with a no-op fallback.

The `Tracker` protocol keeps `pipeline.py` decoupled from MLflow itself.
In production we use `MLflowTracker`; in tests and offline runs we use
`NoOpTracker`. If `mlflow` is not installed we fall back to no-op even
when tracking is enabled in the config, with a logged warning.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


logger = logging.getLogger("bi.tracking")


class Tracker:
    """Minimal tracking interface used by the pipeline."""

    def start_run(self, run_name: str) -> str:
        raise NotImplementedError

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def log_artifact(self, local_path: str | Path) -> None:
        raise NotImplementedError

    def end_run(self, status: str = "FINISHED") -> None:
        raise NotImplementedError


class NoOpTracker(Tracker):
    """Drops every call on the floor; used in tests and `--no-tracking` mode."""

    def start_run(self, run_name: str) -> str:
        return "no-op"

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, local_path: str | Path) -> None:
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        pass


class MLflowTracker(Tracker):
    """Thin wrapper around `mlflow` that uses string keys/values defensively.

    MLflow has hard limits on param value lengths and on metric value types,
    so we cast values to strings (for params) and floats (for metrics) and
    truncate over-long values rather than raising mid-run.
    """

    _MAX_PARAM_LEN = 500   # matches the practical mlflow param-value limit

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
    ):
        import mlflow  # local import so the dep is optional

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._active_run = None

    def start_run(self, run_name: str) -> str:
        self._active_run = self._mlflow.start_run(run_name=run_name)
        return self._active_run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        clean: Dict[str, str] = {}
        for k, v in params.items():
            s = "" if v is None else str(v)
            if len(s) > self._MAX_PARAM_LEN:
                s = s[: self._MAX_PARAM_LEN - 3] + "..."
            clean[str(k)] = s
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            self._mlflow.log_metric(str(k), f, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        self._mlflow.log_artifact(str(local_path))

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run is not None:
            self._mlflow.end_run(status=status)
            self._active_run = None


def build_tracker(config: Dict[str, Any]) -> Tracker:
    """Build the right Tracker based on the `tracking` config block."""
    tracking_cfg = config.get("tracking", {}) or {}
    if not tracking_cfg.get("enabled", True):
        return NoOpTracker()
    try:
        return MLflowTracker(
            experiment_name=str(tracking_cfg.get("experiment_name", "default")),
            tracking_uri=tracking_cfg.get("tracking_uri"),
        )
    except ImportError:
        logger.warning("mlflow is not installed; falling back to NoOpTracker.")
        return NoOpTracker()


@contextmanager
def tracked_run(tracker: Tracker, run_name: str) -> Iterator[str]:
    """Context manager that guarantees `end_run` is called even on failure."""
    run_id = tracker.start_run(run_name)
    try:
        yield run_id
    except Exception:
        tracker.end_run(status="FAILED")
        raise
    else:
        tracker.end_run(status="FINISHED")
