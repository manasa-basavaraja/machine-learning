"""Tests for src.drift: PSI, KS, chi-square, and the orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift import (
    chi_square_categorical,
    detect_drift,
    ks_two_sample,
    psi_categorical,
    psi_numeric,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_psi_numeric_zero_for_identical_distributions():
    rng = _rng(42)
    x = rng.normal(0, 1, size=5000)
    psi = psi_numeric(x, x.copy(), n_bins=10)
    assert psi < 1e-6


def test_psi_numeric_large_for_big_mean_shift():
    rng = _rng(0)
    ref = rng.normal(0, 1, size=5000)
    cur = rng.normal(3, 1, size=5000)
    assert psi_numeric(ref, cur, n_bins=10) > 0.25


def test_psi_numeric_small_for_same_distribution_different_sample():
    rng = _rng(1)
    ref = rng.normal(0, 1, size=5000)
    cur = rng.normal(0, 1, size=5000)
    assert psi_numeric(ref, cur, n_bins=10) < 0.05


def test_ks_small_for_same_distribution():
    rng = _rng(2)
    ref = rng.normal(0, 1, size=5000)
    cur = rng.normal(0, 1, size=5000)
    assert ks_two_sample(ref, cur) < 0.05


def test_ks_large_for_shifted_distribution():
    rng = _rng(3)
    ref = rng.normal(0, 1, size=5000)
    cur = rng.normal(0, 1, size=5000) + 2.0
    assert ks_two_sample(ref, cur) > 0.5


def test_psi_categorical_detects_proportion_shift():
    ref = pd.Series(["A"] * 800 + ["B"] * 200)
    cur = pd.Series(["A"] * 300 + ["B"] * 700)
    assert psi_categorical(ref, cur) > 0.25


def test_psi_categorical_zero_for_identical_proportions():
    ref = pd.Series(["A"] * 800 + ["B"] * 200)
    cur = pd.Series(["A"] * 800 + ["B"] * 200)
    assert psi_categorical(ref, cur) < 1e-6


def test_chi_square_zero_when_proportions_match_exactly():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 500 + ["B"] * 500)
    assert chi_square_categorical(ref, cur) < 1e-6


def test_chi_square_grows_with_shift():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cur = pd.Series(["A"] * 100 + ["B"] * 900)
    assert chi_square_categorical(ref, cur) > 100.0


def test_detect_drift_end_to_end_flags_shifted_features():
    rng = _rng(7)
    ref = pd.DataFrame(
        {
            "x": rng.normal(0, 1, size=3000),
            "cat": rng.choice(["A", "B", "C"], size=3000, p=[0.6, 0.3, 0.1]),
        }
    )
    cur = pd.DataFrame(
        {
            "x": rng.normal(2.0, 1, size=3000),
            "cat": rng.choice(["A", "B", "C"], size=3000, p=[0.6, 0.3, 0.1]),
        }
    )
    config = {
        "binning": {"numeric_bins": 10, "smoothing_epsilon": 1e-4},
        "features": {
            "numeric": [
                {"name": "x", "psi_warn": 0.10, "psi_fail": 0.25, "ks_fail": 0.2},
            ],
            "categorical": [
                {"name": "cat", "psi_warn": 0.10, "psi_fail": 0.25},
            ],
        },
    }
    report = detect_drift(ref, cur, config)

    by_feature = {f.feature: f for f in report.features}
    assert by_feature["x"].status == "fail"
    assert by_feature["cat"].status == "ok"
    assert not report.passed


def test_detect_drift_marks_missing_feature_as_fail():
    ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    config = {
        "features": {
            "numeric": [{"name": "x", "psi_warn": 0.1, "psi_fail": 0.25}]
        }
    }
    report = detect_drift(ref, cur, config)
    assert report.features[0].status == "fail"
    assert "missing" in " ".join(report.features[0].notes)
