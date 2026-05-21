"""Feature engineering for univariate time series forecasting.

Two entry points:

* `build_supervised(series, config)` — turn a series into a fully-labeled
  `(X, y)` dataframe usable by any sklearn regressor. All features are
  built so they are computable from past values *only*, so there is no
  leakage from the target into its own predictors.

* `build_forecast_row(history, target_timestamp, config)` — build the
  single feature row needed to forecast `target_timestamp` given the
  history. Used inside the recursive multi-step forecasting loop in
  `models.RidgeForecaster`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


_TARGET_COL = "__target__"


def _lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {f"lag_{l}": series.shift(l) for l in lags}, index=series.index
    )


def _rolling_mean_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    # shift(1) ensures the rolling window only includes strictly past values.
    return pd.DataFrame(
        {f"rmean_{w}": series.shift(1).rolling(w).mean() for w in windows},
        index=series.index,
    )


def _rolling_std_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {f"rstd_{w}": series.shift(1).rolling(w).std() for w in windows},
        index=series.index,
    )


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dow": index.dayofweek.astype(int),
            "dom": index.day.astype(int),
            "month": index.month.astype(int),
            "weekofyear": index.isocalendar().week.astype(int).values,
            "is_weekend": (index.dayofweek >= 5).astype(int),
        },
        index=index,
    )


def _fourier_features(
    index: pd.DatetimeIndex, period: float, k: int, prefix: str
) -> pd.DataFrame:
    """Add `k` pairs of sin/cos terms with the given period (in days).

    A common, dependency-free way to inject smooth seasonality of arbitrary
    period into a linear model.
    """
    t = (index - index[0]).days.values.astype(float)
    cols: Dict[str, np.ndarray] = {}
    for i in range(1, k + 1):
        cols[f"{prefix}_sin_{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"{prefix}_cos_{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols, index=index)


def _assemble_feature_frame(
    series: pd.Series, feature_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Compose all configured feature blocks into one DataFrame (no target)."""
    blocks: List[pd.DataFrame] = []

    lags = list(feature_cfg.get("lags") or [])
    if lags:
        blocks.append(_lag_features(series, lags))

    rmeans = list(feature_cfg.get("rolling_means") or [])
    if rmeans:
        blocks.append(_rolling_mean_features(series, rmeans))

    rstds = list(feature_cfg.get("rolling_stds") or [])
    if rstds:
        blocks.append(_rolling_std_features(series, rstds))

    if feature_cfg.get("calendar"):
        blocks.append(_calendar_features(series.index))

    fourier_cfg = feature_cfg.get("fourier_seasonality") or {}
    if fourier_cfg.get("yearly"):
        blocks.append(_fourier_features(series.index, 365.25, int(fourier_cfg["yearly"]), "fy"))
    if fourier_cfg.get("weekly"):
        blocks.append(_fourier_features(series.index, 7.0, int(fourier_cfg["weekly"]), "fw"))

    if not blocks:
        return pd.DataFrame(index=series.index)
    return pd.concat(blocks, axis=1)


def build_supervised(
    series: pd.Series, feature_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Return a frame with feature columns + `__target__`, NaNs dropped.

    The dropped rows are the early ones whose lag / rolling features cannot
    be computed from history (e.g. the first `max(lag)` rows).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex.")
    features = _assemble_feature_frame(series, feature_cfg)
    features[_TARGET_COL] = series.values
    features = features.dropna()
    return features


def feature_columns(feature_cfg: Dict[str, Any]) -> List[str]:
    """Return the ordered list of feature column names for a given config.

    Useful when callers want a stable column order without round-tripping
    through `build_supervised`.
    """
    dummy_index = pd.date_range("2024-01-01", periods=400, freq="D")
    dummy = pd.Series(np.arange(400, dtype=float), index=dummy_index)
    frame = _assemble_feature_frame(dummy, feature_cfg)
    return list(frame.columns)


def build_forecast_row(
    history: pd.Series,
    target_timestamp: pd.Timestamp,
    feature_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Build the single feature row needed to forecast `target_timestamp`.

    `history` must already contain all values up to (but not including)
    `target_timestamp` so lag/rolling features are computable. The function
    appends a placeholder NaN at `target_timestamp` then runs the same
    feature builders used in training and returns just the target row.
    """
    if target_timestamp in history.index:
        raise ValueError("target_timestamp must be strictly after history.")
    extended_index = history.index.append(pd.DatetimeIndex([target_timestamp]))
    extended_values = np.concatenate([history.values, [np.nan]])
    extended = pd.Series(extended_values, index=extended_index, name=history.name)

    features = _assemble_feature_frame(extended, feature_cfg)
    row = features.loc[[target_timestamp]]
    return row
