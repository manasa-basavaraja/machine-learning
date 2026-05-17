"""Tests for the model factory and end-to-end pipeline smoke."""

from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset, split_features_target
from src.model import SUPPORTED_MODELS, build_pipeline


_BASE_CONFIG = {
    "seed": 0,
    "data": {
        "source": "synthetic",
        "target": "churn",
        "test_size": 0.2,
        "synthetic": {"n_samples": 400, "churn_rate": 0.3},
    },
    "features": {
        "numeric": [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "num_support_tickets",
        ],
        "categorical": [
            "contract_type",
            "payment_method",
            "internet_service",
            "has_phone_service",
            "paperless_billing",
        ],
    },
}


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_pipeline_fits_and_predicts(model_name: str) -> None:
    df = load_dataset(_BASE_CONFIG)
    X, y = split_features_target(df, _BASE_CONFIG["data"]["target"])

    pipeline = build_pipeline(
        model_name=model_name,
        numeric_features=_BASE_CONFIG["features"]["numeric"],
        categorical_features=_BASE_CONFIG["features"]["categorical"],
        params={},
        seed=0,
    )
    assert isinstance(pipeline, Pipeline)

    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError):
        build_pipeline(
            model_name="xgboost_v999",
            numeric_features=["tenure_months"],
            categorical_features=["contract_type"],
            params={},
            seed=0,
        )
