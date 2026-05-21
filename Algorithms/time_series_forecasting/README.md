# Time Series Forecasting with Walk-Forward Backtesting

A forecasting pipeline that does the things tabular ML pipelines often
get wrong on temporal data: build features without leaking the future,
evaluate with walk-forward (expanding-window) backtesting instead of
a random split, and report forecast quality with the metrics
practitioners actually compare on (MAPE, sMAPE, WAPE, MASE).

## Why this exists

Most "forecast in 30 lines" examples in the wild make one of these
mistakes:

1. **Random train/test split**, which leaks future information into
   training. The right protocol on time series is *walk-forward*:
   train on `[0..t)`, forecast `[t..t+h)`, slide forward, repeat.
2. **Rolling features computed across the full series before splitting**,
   which also leaks. This project builds lag and rolling features inside
   the train window only, then carries the last few historical points
   forward at inference time so prediction features are constructible
   from history alone.
3. **Single accuracy metric.** MAE alone is unit-dependent and can mask
   bad behavior at low-volume periods. The pipeline reports MAE / RMSE
   / MAPE / sMAPE / WAPE / MASE so the user can spot scale, percentage,
   and seasonal-naive-relative views of error in one shot.

## Project layout

```
time_series_forecasting/
├── config/
│   └── forecast.yaml           # data source, features, model, backtest, horizon
├── src/
│   ├── data.py                 # synthetic daily series generator + CSV loader
│   ├── features.py             # lag / rolling / calendar features (leak-safe)
│   ├── models.py               # Naive, SeasonalNaive, MovingAverage, RidgeForecaster
│   ├── backtest.py             # walk-forward CV + per-fold + aggregate metrics
│   ├── metrics.py              # MAE, RMSE, MAPE, sMAPE, WAPE, MASE
│   ├── pipeline.py             # orchestrator: backtest -> final fit -> forecast
│   ├── cli.py                  # CLI entry point
│   └── utils.py                # config loader, logger, seeding
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_metrics.py
├── artifacts/                  # generated at runtime (forecast, metrics, plot data)
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

python -m src.cli --config config/forecast.yaml
```

This will:

1. Load (or synthesize) a daily sales series with weekly + yearly
   seasonality and a mild upward trend.
2. Run walk-forward backtesting with the configured model and horizon.
3. Fit the final model on the full history and emit a forward forecast.
4. Write `artifacts/backtest_metrics.json`, `artifacts/forecast.csv`,
   and `artifacts/run_summary.json`.

## Configuration

```yaml
seed: 42

data:
  source: synthetic            # synthetic | csv
  csv_path: data/series.csv
  date_column: date
  value_column: y
  freq: D                      # pandas frequency string
  synthetic:
    n_days: 730
    trend: 0.02                # per-day drift
    weekly_amplitude: 8.0
    yearly_amplitude: 12.0
    noise_std: 2.0
    baseline: 50.0

features:
  lags: [1, 7, 14, 28]
  rolling_means: [7, 14, 28]
  rolling_stds: [7, 28]
  calendar: true
  fourier_seasonality:
    yearly: 3                  # K = number of Fourier pairs
    weekly: 2

model:
  name: ridge                  # naive | seasonal_naive | moving_average | ridge
  params:
    alpha: 1.0
  season_length: 7             # used by seasonal_naive + MASE
  ma_window: 7                 # used by moving_average

backtest:
  initial_train_size: 365
  horizon: 14                  # forecast h steps each fold
  step: 7                      # slide window forward by this many steps

forecast:
  horizon: 28                  # final forward forecast horizon

artifacts:
  dir: artifacts
```

## Metrics

| Metric | What it answers |
| ------ | --------------- |
| MAE    | Average absolute error in original units. |
| RMSE   | Penalizes large errors more than MAE. |
| MAPE   | Average relative error as a percentage. |
| sMAPE  | Symmetric MAPE that's bounded in `[0, 200]`. |
| WAPE   | Volume-weighted percentage error (good for low-volume periods). |
| MASE   | Error relative to a one-step naive on the training window; `< 1` beats naive. |

## Tests

```bash
pytest tests/ -v
```

Tests cover feature builders (no NaN leakage, correct lag alignment),
each forecaster (naive returns last value, seasonal-naive returns
correct lag, Ridge fits and predicts a smooth series), the walk-forward
splitter (correct fold count, no overlap between train and test), and
each metric (sanity values on hand-crafted inputs, scale invariance
where applicable, MASE behavior on a known baseline).
