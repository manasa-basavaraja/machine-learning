"""Data loading + synthetic daily series generation.

The synthetic generator makes a realistic daily series with a mild
upward trend, weekly seasonality (day-of-week), yearly seasonality, and
gaussian noise. It exists so the pipeline runs end-to-end on any machine
without an external download, and so tests are deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _generate_synthetic(
    n_days: int,
    baseline: float,
    trend: float,
    weekly_amplitude: float,
    yearly_amplitude: float,
    noise_std: float,
    seed: int,
    freq: str = "D",
) -> pd.Series:
    """Build a synthetic series indexed by a DatetimeIndex.

    Composition: `baseline + trend * t + weekly + yearly + noise`. Negative
    values are clipped to 0 since the use case (sales / demand) is non-negative.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq=freq)
    t = np.arange(n_days, dtype=float)

    weekly = weekly_amplitude * np.sin(2 * np.pi * t / 7.0)
    yearly = yearly_amplitude * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0.0, noise_std, size=n_days)

    values = baseline + trend * t + weekly + yearly + noise
    values = np.clip(values, 0.0, None)
    return pd.Series(values, index=dates, name="y")


def load_series(config: Dict[str, Any]) -> pd.Series:
    """Load a univariate time series according to the `data` config block."""
    data_cfg = config["data"]
    source = data_cfg.get("source", "synthetic")

    if source == "synthetic":
        syn = data_cfg.get("synthetic", {})
        return _generate_synthetic(
            n_days=int(syn.get("n_days", 730)),
            baseline=float(syn.get("baseline", 50.0)),
            trend=float(syn.get("trend", 0.02)),
            weekly_amplitude=float(syn.get("weekly_amplitude", 8.0)),
            yearly_amplitude=float(syn.get("yearly_amplitude", 12.0)),
            noise_std=float(syn.get("noise_std", 2.0)),
            seed=int(config.get("seed", 42)),
            freq=str(data_cfg.get("freq", "D")),
        )

    if source == "csv":
        csv_path = Path(data_cfg["csv_path"])
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
        date_col = data_cfg.get("date_column", "date")
        value_col = data_cfg.get("value_column", "y")
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        if value_col not in df.columns:
            raise KeyError(
                f"value_column {value_col!r} not in CSV. Found: {list(df.columns)}"
            )
        series = df[value_col].astype(float)
        series.name = value_col
        return series

    raise ValueError(f"Unknown data.source: {source!r}")
