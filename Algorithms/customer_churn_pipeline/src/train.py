"""Training entry point: load data, tune, fit, evaluate, persist artifacts.

Run from the project root:

    python -m src.train --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .data_loader import load_dataset, split_features_target
from .model import build_pipeline, suggest_params
from .preprocessing import feature_lists_from_config
from .utils import ensure_dir, get_logger, load_config, set_seed


# Optuna is chatty by default; bump it down so training logs stay readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config.",
    )
    return parser.parse_args()


def _tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    features: Dict[str, list],
    logger,
) -> Dict[str, Any]:
    """Run an Optuna study and return the best hyperparameters found."""
    tuning_cfg = config["tuning"]
    model_name = config["model"]["name"]
    seed = int(config["seed"])
    cv_folds = int(config["evaluation"]["cv_folds"])
    scoring = tuning_cfg.get("scoring", "roc_auc")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        pipeline = build_pipeline(
            model_name=model_name,
            numeric_features=features["numeric"],
            categorical_features=features["categorical"],
            params=params,
            seed=seed,
        )
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=int(tuning_cfg.get("n_trials", 25)),
        timeout=tuning_cfg.get("timeout_seconds"),
        show_progress_bar=False,
    )

    logger.info(
        "Optuna best %s=%.4f with params=%s",
        scoring, study.best_value, study.best_params,
    )
    return dict(study.best_params)


def _evaluate(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    """Compute a standard set of binary-classification metrics on a held-out set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "positive_rate": float(np.mean(preds)),
        "decision_threshold": float(threshold),
    }


def _expanded_feature_names(pipeline) -> list:
    """Best-effort retrieval of the post-transform feature names."""
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:  # noqa: BLE001 - older sklearn versions
        return []


def _split_train_test(
    X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["seed"]),
        stratify=y,
    )


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    artifacts_dir = ensure_dir(Path(config["artifacts"]["dir"]))
    log_path = artifacts_dir / config["artifacts"]["log_filename"]
    logger = get_logger("churn.train", log_file=log_path)

    logger.info("Loading dataset (source=%s)", config["data"]["source"])
    df = load_dataset(config)
    logger.info("Loaded %d rows, %d cols", df.shape[0], df.shape[1])

    X, y = split_features_target(df, config["data"]["target"])
    logger.info("Positive class rate: %.3f", float(y.mean()))

    features = feature_lists_from_config(config)
    X_train, X_test, y_train, y_test = _split_train_test(X, y, config)

    if config["tuning"]["enabled"]:
        logger.info("Starting hyperparameter tuning")
        best_params = _tune_hyperparameters(
            X_train, y_train, config, features, logger
        )
    else:
        best_params = {}
        logger.info("Tuning disabled; using estimator defaults")

    final_pipeline = build_pipeline(
        model_name=config["model"]["name"],
        numeric_features=features["numeric"],
        categorical_features=features["categorical"],
        params=best_params,
        seed=seed,
    )

    cv = StratifiedKFold(
        n_splits=int(config["evaluation"]["cv_folds"]),
        shuffle=True,
        random_state=seed,
    )
    cv_scores = cross_val_score(
        final_pipeline, X_train, y_train,
        cv=cv, scoring=config["tuning"]["scoring"], n_jobs=-1,
    )
    logger.info(
        "CV %s: mean=%.4f  std=%.4f",
        config["tuning"]["scoring"], float(np.mean(cv_scores)), float(np.std(cv_scores)),
    )

    logger.info("Fitting final model on train split")
    final_pipeline.fit(X_train, y_train)

    test_metrics = _evaluate(
        final_pipeline, X_test, y_test,
        threshold=float(config["evaluation"]["decision_threshold"]),
    )
    logger.info("Held-out metrics: %s", test_metrics)

    model_path = artifacts_dir / config["artifacts"]["model_filename"]
    metrics_path = artifacts_dir / config["artifacts"]["metrics_filename"]

    joblib.dump(final_pipeline, model_path)
    logger.info("Saved model -> %s", model_path)

    metrics_payload = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_name": config["model"]["name"],
        "best_params": best_params,
        "cv": {
            "scoring": config["tuning"]["scoring"],
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
            "folds": [float(s) for s in cv_scores],
        },
        "test_metrics": test_metrics,
        "feature_names_out": _expanded_feature_names(final_pipeline),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
