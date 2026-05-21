"""Forecasters.

A `Forecaster` is anything with `.fit(series) -> self` and
`.predict(horizon) -> np.ndarray` returning the next `horizon` values.
This keeps the backtest loop trivial and lets baselines (Naive,
SeasonalNaive, MovingAverage) sit next to a learned model (Ridge) under
one interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import build_forecast_row, build_supervised


class Forecaster:
    """Abstract forecaster interface."""

    def fit(self, series: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class NaiveForecaster(Forecaster):
    """Repeat the last observed value for the entire horizon."""

    def __init__(self) -> None:
        self._last: Optional[float] = None

    def fit(self, series: pd.Series) -> "NaiveForecaster":
        if len(series) == 0:
            raise ValueError("Cannot fit NaiveForecaster on empty series.")
        self._last = float(series.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("Call fit before predict.")
        return np.full(horizon, self._last, dtype=float)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value from `season_length` steps ago, cycling forward."""

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = int(season_length)
        self._tail: Optional[np.ndarray] = None

    def fit(self, series: pd.Series) -> "SeasonalNaiveForecaster":
        if len(series) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations to fit "
                f"SeasonalNaiveForecaster; got {len(series)}."
            )
        self._tail = series.iloc[-self.season_length :].to_numpy(dtype=float)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._tail is None:
            raise RuntimeError("Call fit before predict.")
        reps = int(np.ceil(horizon / self.season_length))
        return np.tile(self._tail, reps)[:horizon]


class MovingAverageForecaster(Forecaster):
    """Mean of the last `window` observations, repeated for the horizon."""

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._mean: Optional[float] = None

    def fit(self, series: pd.Series) -> "MovingAverageForecaster":
        if len(series) < self.window:
            raise ValueError(
                f"Need at least {self.window} observations; got {len(series)}."
            )
        self._mean = float(series.iloc[-self.window :].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit before predict.")
        return np.full(horizon, self._mean, dtype=float)


class RidgeForecaster(Forecaster):
    """Ridge regression on engineered features with recursive multi-step prediction.

    At each step of the horizon the latest predicted value is appended to a
    rolling "history" buffer, then `features.build_forecast_row` rebuilds
    the next feature row from that history. This is the standard recursive
    multi-step strategy for tabular models on time series.
    """

    def __init__(
        self,
        feature_cfg: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        freq: str = "D",
    ) -> None:
        self.feature_cfg = dict(feature_cfg)
        self.params = dict(params or {})
        self.freq = freq
        self._model: Optional[Ridge] = None
        self._history: Optional[pd.Series] = None
        self._feature_order: Optional[list] = None

    def fit(self, series: pd.Series) -> "RidgeForecaster":
        frame = build_supervised(series, self.feature_cfg)
        if frame.empty:
            raise ValueError(
                "Not enough history after feature engineering to fit RidgeForecaster."
            )
        feature_cols = [c for c in frame.columns if c != "__target__"]
        self._feature_order = feature_cols
        X = frame[feature_cols].to_numpy(dtype=float)
        y = frame["__target__"].to_numpy(dtype=float)

        self._model = Ridge(**self.params)
        self._model.fit(X, y)
        self._history = series.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None or self._history is None or self._feature_order is None:
            raise RuntimeError("Call fit before predict.")

        history = self._history.copy()
        offset = pd.tseries.frequencies.to_offset(self.freq)
        out = np.empty(horizon, dtype=float)
        for step in range(horizon):
            next_ts = history.index[-1] + offset
            row = build_forecast_row(history, next_ts, self.feature_cfg)
            x = row.reindex(columns=self._feature_order).to_numpy(dtype=float)
            if np.isnan(x).any():
                # In rare configs (very long lags + tiny history) features
                # can still contain NaN; fall back to the last observation
                # rather than crashing the whole run.
                yhat = float(history.iloc[-1])
            else:
                yhat = float(self._model.predict(x)[0])
            out[step] = yhat
            history = pd.concat([history, pd.Series([yhat], index=[next_ts])])
        return out


def build_forecaster(model_cfg: Dict[str, Any], feature_cfg: Dict[str, Any], freq: str) -> Forecaster:
    """Instantiate a Forecaster from the `model` block of the config."""
    name = str(model_cfg.get("name", "ridge")).lower()
    params = model_cfg.get("params") or {}

    if name == "naive":
        return NaiveForecaster()
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(season_length=int(model_cfg.get("season_length", 7)))
    if name == "moving_average":
        return MovingAverageForecaster(window=int(model_cfg.get("ma_window", 7)))
    if name == "ridge":
        return RidgeForecaster(feature_cfg=feature_cfg, params=params, freq=freq)
    raise ValueError(
        f"Unknown model.name {name!r}. Choose from: naive, seasonal_naive, "
        "moving_average, ridge."
    )
