"""Shared fixtures for the time series tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def daily_series() -> pd.Series:
    """A deterministic 200-day series with linear trend + weekly seasonality."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    t = np.arange(200, dtype=float)
    values = 50.0 + 0.05 * t + 5.0 * np.sin(2 * np.pi * t / 7.0) + rng.normal(0, 0.5, 200)
    return pd.Series(values, index=idx, name="y")


@pytest.fixture()
def short_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.Series(np.arange(30, dtype=float), index=idx, name="y")


@pytest.fixture()
def feature_config_small() -> dict:
    return {
        "lags": [1, 7],
        "rolling_means": [7],
        "rolling_stds": [7],
        "calendar": True,
        "fourier_seasonality": {"weekly": 2},
    }
