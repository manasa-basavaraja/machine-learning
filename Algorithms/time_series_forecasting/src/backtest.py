"""Walk-forward (expanding-window) backtesting.

For a series of length N, the backtester:
1. Trains on `series[:initial_train_size]` and forecasts the next `horizon`.
2. Slides the training window forward by `step`, repeats.
3. Continues until there isn't enough remaining data for a full `horizon`.

Returns per-fold metrics plus an aggregate (mean across folds), which is
the right granularity for production model selection: a model that
collapses on a single fold gets surfaced instead of getting averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from .metrics import compute_all
from .models import Forecaster


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: Dict[str, float]
    n_train: int
    n_test: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "train_start": str(self.train_start.date()) if hasattr(self.train_start, "date") else str(self.train_start),
            "train_end": str(self.train_end.date()) if hasattr(self.train_end, "date") else str(self.train_end),
            "test_start": str(self.test_start.date()) if hasattr(self.test_start, "date") else str(self.test_start),
            "test_end": str(self.test_end.date()) if hasattr(self.test_end, "date") else str(self.test_end),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
        }


@dataclass
class BacktestResult:
    folds: List[FoldResult] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    horizon: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "n_folds": len(self.folds),
            "aggregate": {k: float(v) for k, v in self.aggregate.items()},
            "folds": [f.to_dict() for f in self.folds],
        }


def _generate_split_indices(
    n: int, initial_train_size: int, horizon: int, step: int
) -> List[int]:
    """Return the list of split positions `t` where `series[:t]` is train.

    `t` ranges from `initial_train_size` to `n - horizon` in `step` strides
    so every fold has a full `horizon` of held-out data.
    """
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if n < initial_train_size + horizon:
        return []
    return list(range(initial_train_size, n - horizon + 1, step))


def walk_forward(
    series: pd.Series,
    forecaster_factory: Callable[[], Forecaster],
    initial_train_size: int,
    horizon: int,
    step: int = 1,
    mase_season_length: int = 1,
) -> BacktestResult:
    """Run walk-forward backtesting and return per-fold + aggregate metrics.

    A fresh Forecaster is built per fold via `forecaster_factory()` so model
    state from a previous fold can never bleed into the next.
    """
    splits = _generate_split_indices(len(series), initial_train_size, horizon, step)
    folds: List[FoldResult] = []

    for fold_idx, t in enumerate(splits):
        train = series.iloc[:t]
        test = series.iloc[t : t + horizon]

        forecaster = forecaster_factory()
        forecaster.fit(train)
        preds = np.asarray(forecaster.predict(horizon), dtype=float)

        metrics = compute_all(
            y_true=test.to_numpy(dtype=float),
            y_pred=preds,
            training_series=train.to_numpy(dtype=float),
            season_length=mase_season_length,
        )
        folds.append(FoldResult(
            fold=fold_idx,
            train_start=train.index[0],
            train_end=train.index[-1],
            test_start=test.index[0],
            test_end=test.index[-1],
            metrics=metrics,
            n_train=int(len(train)),
            n_test=int(len(test)),
        ))

    aggregate: Dict[str, float] = {}
    if folds:
        metric_keys = sorted({k for f in folds for k in f.metrics.keys()})
        for k in metric_keys:
            values = [f.metrics[k] for f in folds if not np.isnan(f.metrics.get(k, np.nan))]
            aggregate[k] = float(np.mean(values)) if values else float("nan")

    return BacktestResult(folds=folds, aggregate=aggregate, horizon=horizon)
