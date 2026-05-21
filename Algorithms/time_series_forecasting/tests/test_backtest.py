"""Tests for src.backtest."""

from __future__ import annotations

import math

import pytest

from src.backtest import _generate_split_indices, walk_forward
from src.models import NaiveForecaster, RidgeForecaster, SeasonalNaiveForecaster


def test_split_indices_step_one(daily_series):
    indices = _generate_split_indices(
        n=len(daily_series), initial_train_size=100, horizon=10, step=1
    )
    # last fold starts at n - horizon = 190
    assert indices[0] == 100
    assert indices[-1] == 190
    assert all(b - a == 1 for a, b in zip(indices, indices[1:]))


def test_split_indices_returns_empty_when_not_enough_data():
    indices = _generate_split_indices(n=50, initial_train_size=100, horizon=10, step=1)
    assert indices == []


def test_split_indices_validates_arguments():
    with pytest.raises(ValueError):
        _generate_split_indices(n=100, initial_train_size=0, horizon=1, step=1)
    with pytest.raises(ValueError):
        _generate_split_indices(n=100, initial_train_size=10, horizon=0, step=1)
    with pytest.raises(ValueError):
        _generate_split_indices(n=100, initial_train_size=10, horizon=1, step=0)


def test_walk_forward_no_overlap_between_train_and_test(daily_series):
    result = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: NaiveForecaster(),
        initial_train_size=100,
        horizon=14,
        step=7,
    )
    for fold in result.folds:
        assert fold.train_end < fold.test_start


def test_walk_forward_each_fold_has_full_horizon(daily_series):
    result = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: NaiveForecaster(),
        initial_train_size=100,
        horizon=14,
        step=7,
    )
    for fold in result.folds:
        assert fold.n_test == 14


def test_walk_forward_aggregate_matches_per_fold_mean(daily_series):
    result = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: NaiveForecaster(),
        initial_train_size=100,
        horizon=14,
        step=7,
        mase_season_length=7,
    )
    if not result.folds:
        pytest.skip("Not enough data for any folds.")
    expected_mae = sum(f.metrics["mae"] for f in result.folds) / len(result.folds)
    assert result.aggregate["mae"] == pytest.approx(expected_mae)


def test_walk_forward_ridge_beats_naive_on_seasonal_data(daily_series, feature_config_small):
    naive_res = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: NaiveForecaster(),
        initial_train_size=120,
        horizon=14,
        step=7,
    )
    ridge_res = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: RidgeForecaster(
            feature_cfg=feature_config_small, params={"alpha": 1.0}
        ),
        initial_train_size=120,
        horizon=14,
        step=7,
    )
    if not naive_res.folds or not ridge_res.folds:
        pytest.skip("Not enough data for both backtests.")
    # Ridge should typically beat naive on a clearly seasonal+trended series.
    assert ridge_res.aggregate["mae"] < naive_res.aggregate["mae"]


def test_walk_forward_seasonal_naive_factory_produces_independent_models(daily_series):
    """Each fold must call the factory; mutating one forecaster must not affect others."""
    seen = []

    def factory():
        f = SeasonalNaiveForecaster(season_length=7)
        seen.append(f)
        return f

    result = walk_forward(
        series=daily_series,
        forecaster_factory=factory,
        initial_train_size=100,
        horizon=7,
        step=7,
    )
    assert len(seen) == len(result.folds)
    assert all(a is not b for i, a in enumerate(seen) for b in seen[i + 1 :])


def test_walk_forward_metrics_are_finite(daily_series):
    result = walk_forward(
        series=daily_series,
        forecaster_factory=lambda: NaiveForecaster(),
        initial_train_size=100,
        horizon=14,
        step=14,
        mase_season_length=7,
    )
    for fold in result.folds:
        for k, v in fold.metrics.items():
            assert math.isfinite(v), f"metric {k} not finite on fold {fold.fold}"
