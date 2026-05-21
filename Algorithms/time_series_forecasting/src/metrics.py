"""Forecast accuracy metrics.

All functions are pure NumPy and accept array-likes. They handle NaNs by
dropping pairwise (any NaN in either y_true or y_pred at index i excludes
that row from the metric).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _align(y_true, y_pred) -> tuple:
    """Convert to arrays, validate shape, drop pairwise NaNs."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-9) -> float:
    """Mean absolute percentage error (×100). Zeros in y_true are skipped."""
    yt, yp = _align(y_true, y_pred)
    nonzero = np.abs(yt) > epsilon
    if not nonzero.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))


def smape(y_true, y_pred, epsilon: float = 1e-9) -> float:
    """Symmetric MAPE (×100), bounded in [0, 200]."""
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom > epsilon
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]))


def wape(y_true, y_pred, epsilon: float = 1e-9) -> float:
    """Weighted APE: sum(|err|) / sum(|y_true|) × 100."""
    yt, yp = _align(y_true, y_pred)
    denom = float(np.sum(np.abs(yt)))
    if denom < epsilon:
        return float("nan")
    return float(100.0 * float(np.sum(np.abs(yt - yp))) / denom)


def mase(
    y_true,
    y_pred,
    training_series: np.ndarray,
    season_length: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample MAE of a naive seasonal forecaster on the
    *training* series. Values `< 1` beat that naive baseline. Common in
    forecasting competitions because it is scale-free and interpretable.
    """
    yt, yp = _align(y_true, y_pred)
    training = np.asarray(training_series, dtype=float).ravel()
    training = training[~np.isnan(training)]
    if training.size <= season_length:
        return float("nan")
    naive_diffs = np.abs(training[season_length:] - training[:-season_length])
    scale = float(naive_diffs.mean())
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / scale)


def compute_all(
    y_true,
    y_pred,
    training_series: Optional[np.ndarray] = None,
    season_length: int = 1,
) -> Dict[str, float]:
    """Return a single dict containing every metric in this module."""
    results = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
    if training_series is not None:
        results["mase"] = mase(y_true, y_pred, training_series, season_length=season_length)
    return results
