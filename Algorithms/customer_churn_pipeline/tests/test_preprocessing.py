"""Tests for the preprocessing ColumnTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_preprocessor


@pytest.fixture()
def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure_months": [1, 12, np.nan, 60],
            "monthly_charges": [29.9, 80.5, 55.0, np.nan],
            "contract_type": ["Month-to-month", "One year", None, "Two year"],
            "payment_method": ["Mailed check", "Credit card", "Mailed check", None],
        }
    )


def test_preprocessor_imputes_and_encodes(toy_frame: pd.DataFrame) -> None:
    pre = build_preprocessor(
        numeric_features=["tenure_months", "monthly_charges"],
        categorical_features=["contract_type", "payment_method"],
    )
    transformed = pre.fit_transform(toy_frame)

    assert transformed.shape[0] == len(toy_frame)
    assert not np.isnan(np.asarray(transformed)).any(), "imputation must remove NaNs"
    # 2 scaled numeric columns + one-hot expansions.
    assert transformed.shape[1] >= 2 + 3 + 3


def test_preprocessor_handles_unseen_category(toy_frame: pd.DataFrame) -> None:
    pre = build_preprocessor(
        numeric_features=["tenure_months", "monthly_charges"],
        categorical_features=["contract_type", "payment_method"],
    )
    pre.fit(toy_frame)

    unseen = pd.DataFrame(
        {
            "tenure_months": [5],
            "monthly_charges": [40.0],
            "contract_type": ["Quarterly"],
            "payment_method": ["Crypto"],
        }
    )
    transformed = pre.transform(unseen)
    assert transformed.shape == (1, pre.transform(toy_frame).shape[1])


def test_preprocessor_drops_unlisted_columns(toy_frame: pd.DataFrame) -> None:
    frame = toy_frame.copy()
    frame["extra_noise"] = ["a", "b", "c", "d"]

    pre = build_preprocessor(
        numeric_features=["tenure_months", "monthly_charges"],
        categorical_features=["contract_type", "payment_method"],
    )
    transformed = pre.fit_transform(frame)

    base_width = build_preprocessor(
        numeric_features=["tenure_months", "monthly_charges"],
        categorical_features=["contract_type", "payment_method"],
    ).fit_transform(toy_frame).shape[1]
    assert transformed.shape[1] == base_width
