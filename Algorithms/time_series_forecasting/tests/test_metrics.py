"""Tests for src.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import compute_all, mae, mape, mase, rmse, smape, wape


def test_mae_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    assert mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0])
    small = np.array([1.0, 1.0, 1.0])
    spiky = np.array([0.0, 0.0, 3.0])
    assert mae(y_true, spiky) == pytest.approx(1.0)
    assert rmse(y_true, spiky) > mae(y_true, spiky)
    assert mae(y_true, small) == rmse(y_true, small)


def test_mape_skips_zero_truths():
    y_true = [0, 100, 200]
    y_pred = [50, 110, 220]
    # Zero is skipped; mean of (10/100, 20/200) = 0.10 -> 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_smape_is_symmetric():
    a = [100, 100, 100]
    b = [110, 110, 110]
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_smape_bounded_by_200():
    assert smape([1, 1, 1], [1000, 1000, 1000]) <= 200.0


def test_wape_weighted_by_volume():
    y_true = [10, 100]
    y_pred = [9, 90]
    assert wape(y_true, y_pred) == pytest.approx(100.0 * 11 / 110)


def test_mase_one_for_naive_predictions():
    training = np.arange(1, 50, dtype=float)
    y_true = np.arange(50, 60, dtype=float)
    y_pred_naive = np.full(10, training[-1])
    val = mase(y_true, y_pred_naive, training_series=training, season_length=1)
    assert val > 0.0
    assert math.isfinite(val)


def test_mase_zero_for_perfect_predictions():
    training = np.arange(1, 50, dtype=float)
    y_true = np.arange(50, 60, dtype=float)
    assert mase(y_true, y_true, training_series=training, season_length=1) == 0.0


def test_compute_all_returns_expected_keys():
    training = np.arange(1, 30, dtype=float)
    y_true = np.arange(30, 40, dtype=float)
    y_pred = y_true + 1
    out = compute_all(y_true, y_pred, training_series=training, season_length=1)
    assert {"mae", "rmse", "mape", "smape", "wape", "mase"} == set(out.keys())
    assert all(math.isfinite(v) for v in out.values())


def test_metrics_drop_pairwise_nans():
    y_true = [1.0, np.nan, 3.0]
    y_pred = [1.0, 5.0, 3.0]
    assert mae(y_true, y_pred) == 0.0
