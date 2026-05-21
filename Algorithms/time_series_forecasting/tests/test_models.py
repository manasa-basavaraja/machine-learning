"""Tests for src.models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import (
    MovingAverageForecaster,
    NaiveForecaster,
    RidgeForecaster,
    SeasonalNaiveForecaster,
    build_forecaster,
)


def test_naive_returns_last_value(daily_series):
    f = NaiveForecaster().fit(daily_series)
    preds = f.predict(horizon=5)
    assert preds.shape == (5,)
    assert np.allclose(preds, daily_series.iloc[-1])


def test_naive_requires_fit_before_predict():
    with pytest.raises(RuntimeError):
        NaiveForecaster().predict(3)


def test_seasonal_naive_repeats_last_season(daily_series):
    f = SeasonalNaiveForecaster(season_length=7).fit(daily_series)
    preds = f.predict(horizon=14)
    expected_one_period = daily_series.iloc[-7:].to_numpy()
    assert np.allclose(preds[:7], expected_one_period)
    assert np.allclose(preds[7:14], expected_one_period)


def test_seasonal_naive_rejects_short_history():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(season_length=7).fit(short)


def test_moving_average_returns_mean_of_window(daily_series):
    f = MovingAverageForecaster(window=10).fit(daily_series)
    preds = f.predict(horizon=3)
    expected = daily_series.iloc[-10:].mean()
    assert np.allclose(preds, expected)


def test_ridge_predict_shape(daily_series, feature_config_small):
    f = RidgeForecaster(feature_cfg=feature_config_small, params={"alpha": 1.0})
    f.fit(daily_series)
    preds = f.predict(horizon=14)
    assert preds.shape == (14,)
    assert np.all(np.isfinite(preds))


def test_ridge_predict_is_close_to_data_range(daily_series, feature_config_small):
    f = RidgeForecaster(feature_cfg=feature_config_small, params={"alpha": 1.0})
    f.fit(daily_series)
    preds = f.predict(horizon=14)
    # Forecasts should sit within a sane band around the recent history range.
    recent = daily_series.iloc[-30:]
    lo, hi = recent.min() - 3 * recent.std(), recent.max() + 3 * recent.std()
    assert preds.min() >= lo
    assert preds.max() <= hi


def test_build_forecaster_dispatch():
    feat = {"lags": [1]}
    assert isinstance(
        build_forecaster({"name": "naive"}, feat, "D"), NaiveForecaster
    )
    assert isinstance(
        build_forecaster({"name": "seasonal_naive", "season_length": 7}, feat, "D"),
        SeasonalNaiveForecaster,
    )
    assert isinstance(
        build_forecaster({"name": "moving_average", "ma_window": 5}, feat, "D"),
        MovingAverageForecaster,
    )
    assert isinstance(
        build_forecaster({"name": "ridge", "params": {"alpha": 0.5}}, feat, "D"),
        RidgeForecaster,
    )


def test_build_forecaster_unknown_name_raises():
    with pytest.raises(ValueError):
        build_forecaster({"name": "xgboost_lstm_9000"}, {"lags": [1]}, "D")
