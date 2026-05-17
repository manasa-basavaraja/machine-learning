# Customer Churn Prediction Pipeline

A production-style, end-to-end binary classification pipeline for predicting
telecom customer churn. The project is structured the way ML/data engineers
typically ship models internally: configuration is externalized, transformations
live inside a single `sklearn` `Pipeline` (so train and inference behave
identically), hyperparameters are tuned with Optuna, and every training run
persists a self-contained model artifact alongside its metrics.

## Why this is structured this way

A common failure mode in real teams is training in a notebook with ad-hoc
preprocessing, then losing the exact preprocessing recipe at inference time
(a.k.a. "training-serving skew"). This project avoids that by:

- Fitting **one** `Pipeline` that owns *all* feature transformations and the
  estimator. The same object is used for `.fit`, evaluation, and `.predict`.
- Persisting that pipeline with `joblib` next to the metrics JSON, so a
  downstream service can `joblib.load` and score immediately.
- Reading every tunable knob from `config/config.yaml` so re-runs and CI jobs
  don't depend on hard-coded values in source files.

## Project layout

```
customer_churn_pipeline/
├── config/
│   └── config.yaml              # data paths, model params, tuning settings
├── src/
│   ├── data_loader.py           # IO + synthetic data generation
│   ├── preprocessing.py         # ColumnTransformer factory
│   ├── model.py                 # Pipeline factory + Optuna search space
│   ├── train.py                 # CLI entry point: train + tune + persist
│   ├── predict.py               # CLI entry point: batch scoring
│   └── utils.py                 # config loader, logger, seed helpers
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── artifacts/                   # created at runtime (model + metrics)
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Train (uses synthetic data if config.data.source == "synthetic")
python -m src.train --config config/config.yaml

# Batch score a CSV
python -m src.predict \
    --config config/config.yaml \
    --model artifacts/model.joblib \
    --input data/new_customers.csv \
    --output artifacts/predictions.csv
```

Run from the project root (`Algorithms/customer_churn_pipeline/`).

## Configuration

All behavior is driven by `config/config.yaml`. Notable sections:

- `data.source`: `synthetic` (generates a realistic Telco-style frame) or
  `csv` (reads `data.csv_path`).
- `model.name`: `logistic_regression` | `random_forest` | `gradient_boosting`.
- `tuning.enabled`: when `true`, runs Optuna with `tuning.n_trials` trials and
  uses the best params for the final fit.
- `evaluation.cv_folds`: stratified K-fold CV folds used during tuning and
  final evaluation.

## Outputs

Each training run writes to `artifacts/`:

- `model.joblib` — fitted `Pipeline` (preprocessing + estimator).
- `metrics.json` — CV and held-out test metrics (ROC-AUC, PR-AUC, F1,
  precision, recall, accuracy), best hyperparameters, and feature names.
- `train.log` — structured log of the run.

## Testing

```bash
pytest tests/ -v
```

Tests cover the preprocessing transformer (shape, no leakage of NaNs) and
the model factory (pipeline composition, smoke-fit on a tiny synthetic set).
