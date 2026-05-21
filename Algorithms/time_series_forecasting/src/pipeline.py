"""End-to-end orchestrator: load -> backtest -> final fit -> forecast -> persist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .backtest import BacktestResult, walk_forward
from .data import load_series
from .models import build_forecaster
from .utils import ensure_dir, get_logger, set_seed


@dataclass
class RunArtifacts:
    metrics_path: Path
    forecast_path: Path
    summary_path: Path
    backtest: BacktestResult
    forecast: pd.Series


def _forecaster_factory(config: Dict[str, Any]):
    """Build a zero-arg factory used by the backtester."""
    model_cfg = config["model"]
    feature_cfg = config["features"]
    freq = str(config["data"].get("freq", "D"))

    def _factory():
        return build_forecaster(model_cfg, feature_cfg, freq)

    return _factory


def _make_future_index(series: pd.Series, horizon: int, freq: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    start = series.index[-1] + offset
    return pd.date_range(start=start, periods=horizon, freq=freq)


def run_pipeline(config: Dict[str, Any]) -> RunArtifacts:
    """Run the full pipeline. Returns the in-memory artifacts and persists them."""
    logger = get_logger()
    set_seed(int(config.get("seed", 42)))

    series = load_series(config)
    logger.info("Loaded series: n=%d  start=%s  end=%s",
                len(series), series.index[0].date(), series.index[-1].date())

    factory = _forecaster_factory(config)
    bt_cfg = config["backtest"]
    season_length = int(config["model"].get("season_length", 7))

    backtest_result = walk_forward(
        series=series,
        forecaster_factory=factory,
        initial_train_size=int(bt_cfg["initial_train_size"]),
        horizon=int(bt_cfg["horizon"]),
        step=int(bt_cfg.get("step", 1)),
        mase_season_length=season_length,
    )
    logger.info(
        "Backtest: n_folds=%d  aggregate=%s",
        len(backtest_result.folds),
        {k: round(v, 4) for k, v in backtest_result.aggregate.items()},
    )

    final_forecaster = factory()
    final_forecaster.fit(series)
    horizon = int(config["forecast"]["horizon"])
    future_index = _make_future_index(series, horizon, str(config["data"].get("freq", "D")))
    forecast_values = final_forecaster.predict(horizon)
    forecast = pd.Series(forecast_values, index=future_index, name="yhat")

    artifacts_cfg = config["artifacts"]
    artifacts_dir = ensure_dir(Path(artifacts_cfg["dir"]))
    metrics_path = artifacts_dir / artifacts_cfg.get("metrics_filename", "backtest_metrics.json")
    forecast_path = artifacts_dir / artifacts_cfg.get("forecast_filename", "forecast.csv")
    summary_path = artifacts_dir / artifacts_cfg.get("summary_filename", "run_summary.json")

    metrics_path.write_text(json.dumps(backtest_result.to_dict(), indent=2), encoding="utf-8")
    forecast.to_frame().to_csv(forecast_path, index_label="date")
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": config["model"],
                "features": config["features"],
                "backtest": {
                    "horizon": backtest_result.horizon,
                    "n_folds": len(backtest_result.folds),
                    "aggregate": backtest_result.aggregate,
                },
                "forecast": {
                    "horizon": horizon,
                    "start": str(future_index[0].date()),
                    "end": str(future_index[-1].date()),
                    "mean": float(forecast.mean()),
                },
                "series": {
                    "n": int(len(series)),
                    "start": str(series.index[0].date()),
                    "end": str(series.index[-1].date()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote metrics=%s  forecast=%s  summary=%s",
                metrics_path, forecast_path, summary_path)

    return RunArtifacts(
        metrics_path=metrics_path,
        forecast_path=forecast_path,
        summary_path=summary_path,
        backtest=backtest_result,
        forecast=forecast,
    )
