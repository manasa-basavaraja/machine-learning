"""Shared fixtures.

A `FakeEstimator` is used everywhere instead of a real sklearn model so the
tests don't depend on sklearn-version-specific behavior and run instantly.
It mimics the bits of the `predict_proba` contract that the scorer cares
about.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


class FakeEstimator:
    """Deterministic stand-in for a sklearn classifier.

    `predict_proba` returns probabilities derived from a hash of each row's
    string representation so the output is stable and varies between rows.
    """

    def __init__(self, feature_columns: List[str] | None = None):
        self.feature_columns = feature_columns

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_columns is not None:
                missing = set(self.feature_columns) - set(X.columns)
                if missing:
                    raise ValueError(f"FakeEstimator missing columns: {missing}")
            rows = X.astype(str).agg("|".join, axis=1).tolist()
        else:
            rows = [str(row) for row in X]
        proba_pos = np.array([(abs(hash(r)) % 1000) / 1000.0 for r in rows], dtype=float)
        return np.column_stack([1.0 - proba_pos, proba_pos])


@pytest.fixture()
def fake_model() -> FakeEstimator:
    return FakeEstimator()


@pytest.fixture()
def toy_chunk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105],
            "tenure_months": [1, 12, 24, 36, 48],
            "monthly_charges": [29.9, 80.5, 55.0, 100.0, 45.0],
            "contract_type": [
                "Month-to-month", "One year", "Two year",
                "Month-to-month", "One year",
            ],
        }
    )
