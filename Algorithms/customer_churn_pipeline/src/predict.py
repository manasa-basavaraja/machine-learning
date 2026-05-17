"""Batch inference entry point.

Loads the persisted Pipeline, scores a CSV, and writes predictions
alongside their probabilities. Designed to be called from a scheduler
(Airflow, cron, etc.) on a new batch of customers.

Usage:
    python -m src.predict \
        --config config/config.yaml \
        --model artifacts/model.joblib \
        --input data/new_customers.csv \
        --output artifacts/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .utils import get_logger, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a CSV with a fitted model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    logger = get_logger("churn.predict")

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config["evaluation"]["decision_threshold"])
    )

    logger.info("Loading model from %s", model_path)
    pipeline = joblib.load(model_path)

    logger.info("Reading input %s", input_path)
    df = pd.read_csv(input_path)

    target = config["data"]["target"]
    if target in df.columns:
        df = df.drop(columns=[target])

    proba = pipeline.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = (proba >= threshold).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d predictions to %s (threshold=%.3f, positive_rate=%.3f)",
        len(out), output_path, threshold, float(out["churn_prediction"].mean()),
    )


if __name__ == "__main__":
    main()
