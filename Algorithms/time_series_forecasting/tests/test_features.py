"""Tests for src.features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_forecast_row, build_supervised, feature_columns


def test_build_supervised_drops_initial_rows_with_nan_features(daily_series, feature_config_small):
    frame = build_supervised(daily_series, feature_config_small)
    assert not frame.isna().any().any(), "feature frame must be NaN-free after dropna"
    # rolling_means=[7] requires lag of 1..7 -> first 7 rows dropped at minimum.
    assert len(frame) <= len(daily_series) - 7


def test_lag_1_equals_previous_value(daily_series, feature_config_small):
    frame = build_supervised(daily_series, feature_config_small)
    first_idx = frame.index[0]
    prev_idx = daily_series.index.get_loc(first_idx) - 1
    assert frame.loc[first_idx, "lag_1"] == pytest.approx(daily_series.iloc[prev_idx])


def test_target_column_matches_series(daily_series, feature_config_small):
    frame = build_supervised(daily_series, feature_config_small)
    for ts, row in frame.iterrows():
        assert row["__target__"] == pytest.approx(daily_series.loc[ts])


def test_rolling_mean_excludes_current_value(daily_series, feature_config_small):
    frame = build_supervised(daily_series, feature_config_small)
    sample_idx = frame.index[20]
    pos = daily_series.index.get_loc(sample_idx)
    expected = daily_series.iloc[pos - 7 : pos].mean()
    assert frame.loc[sample_idx, "rmean_7"] == pytest.approx(expected)


def test_feature_columns_matches_build_supervised(daily_series, feature_config_small):
    frame = build_supervised(daily_series, feature_config_small)
    cols = feature_columns(feature_config_small)
    assert set(cols) == set(frame.columns) - {"__target__"}


def test_build_forecast_row_is_constructible_from_history_alone(daily_series, feature_config_small):
    history = daily_series.iloc[:-1]
    target_ts = daily_series.index[-1]
    row = build_forecast_row(history, target_ts, feature_config_small)
    assert len(row) == 1
    assert not row.isna().any().any()


def test_build_forecast_row_lag_1_is_last_history_value(daily_series, feature_config_small):
    history = daily_series.iloc[:-1]
    target_ts = daily_series.index[-1]
    row = build_forecast_row(history, target_ts, feature_config_small)
    assert row.iloc[0]["lag_1"] == pytest.approx(history.iloc[-1])


def test_build_forecast_row_rejects_target_in_history(daily_series, feature_config_small):
    target_ts = daily_series.index[-1]
    with pytest.raises(ValueError):
        build_forecast_row(daily_series, target_ts, feature_config_small)


def test_calendar_features_present_when_enabled(daily_series):
    cfg = {"lags": [1], "calendar": True}
    frame = build_supervised(daily_series, cfg)
    for col in ["dow", "dom", "month", "weekofyear", "is_weekend"]:
        assert col in frame.columns


def test_fourier_features_added(daily_series):
    cfg = {
        "lags": [1],
        "calendar": False,
        "fourier_seasonality": {"weekly": 2, "yearly": 1},
    }
    cols = feature_columns(cfg)
    assert "fw_sin_1" in cols and "fw_cos_2" in cols
    assert "fy_sin_1" in cols and "fy_cos_1" in cols
