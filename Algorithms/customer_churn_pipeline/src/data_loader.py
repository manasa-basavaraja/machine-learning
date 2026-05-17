"""Data loading + synthetic Telco-style data generation.

The synthetic generator exists so the pipeline is runnable end-to-end on
any machine with no external download. The schema and value distributions
roughly mirror the public Telco Churn dataset, which means the same
preprocessing config works on either source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


_CONTRACT_TYPES = ("Month-to-month", "One year", "Two year")
_PAYMENT_METHODS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card",
)
_INTERNET_SERVICE = ("DSL", "Fiber optic", "None")


def _generate_synthetic(
    n_samples: int,
    churn_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a Telco-style churn dataset with realistic correlations.

    Churn probability is driven primarily by contract type, tenure, and
    monthly charges so that a model actually has signal to learn.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(_CONTRACT_TYPES, size=n_samples, p=[0.55, 0.25, 0.20])
    payment = rng.choice(_PAYMENT_METHODS, size=n_samples)
    internet = rng.choice(_INTERNET_SERVICE, size=n_samples, p=[0.35, 0.45, 0.20])
    has_phone = rng.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    tenure = np.clip(rng.normal(loc=32, scale=24, size=n_samples), 0, 72).astype(int)
    monthly_charges = np.clip(
        rng.normal(loc=65, scale=30, size=n_samples), 18, 120
    ).round(2)
    total_charges = (monthly_charges * np.maximum(tenure, 1) * rng.uniform(
        0.9, 1.05, size=n_samples
    )).round(2)
    support_tickets = rng.poisson(lam=1.2, size=n_samples)

    # Latent score that drives churn, calibrated so the realized rate
    # is close to the requested `churn_rate`.
    contract_risk = np.where(
        contract == "Month-to-month", 1.4,
        np.where(contract == "One year", -0.3, -1.2),
    )
    tenure_risk = -0.025 * tenure
    charge_risk = 0.012 * (monthly_charges - 65)
    ticket_risk = 0.18 * support_tickets
    fiber_risk = np.where(internet == "Fiber optic", 0.5, 0.0)

    logit = contract_risk + tenure_risk + charge_risk + ticket_risk + fiber_risk
    logit += rng.normal(0, 0.6, size=n_samples)

    # Shift the intercept to hit the target churn rate.
    target_logit = np.quantile(logit, 1 - churn_rate)
    prob = 1.0 / (1.0 + np.exp(-(logit - target_logit)))
    churn = (rng.uniform(size=n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": support_tickets,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "has_phone_service": has_phone,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the dataset according to the `data` block of the config."""
    data_cfg = config["data"]
    source = data_cfg.get("source", "synthetic")

    if source == "synthetic":
        syn = data_cfg.get("synthetic", {})
        return _generate_synthetic(
            n_samples=int(syn.get("n_samples", 5000)),
            churn_rate=float(syn.get("churn_rate", 0.27)),
            seed=int(config.get("seed", 42)),
        )

    if source == "csv":
        csv_path = Path(data_cfg["csv_path"])
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
        return pd.read_csv(csv_path)

    raise ValueError(f"Unknown data.source: {source!r}")


def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into (X, y), validating the target exists."""
    if target not in df.columns:
        raise KeyError(
            f"Target column {target!r} not in dataframe. "
            f"Available: {list(df.columns)}"
        )
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y
