"""Model + Optuna search space factory.

The pipeline returned by `build_pipeline` is a single sklearn `Pipeline`
that owns both preprocessing and the estimator, so train/predict can never
diverge. `suggest_params` defines the Optuna search space per estimator.
"""

from __future__ import annotations

from typing import Any, Dict, List

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


SUPPORTED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting")


def _build_estimator(name: str, params: Dict[str, Any], seed: int):
    """Instantiate a bare estimator from a name + params dict."""
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=seed,
            **params,
        )
    raise ValueError(
        f"Unsupported model {name!r}. Choose from: {SUPPORTED_MODELS}"
    )


def build_pipeline(
    model_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
    params: Dict[str, Any],
    seed: int,
) -> Pipeline:
    """Compose preprocessing + estimator into a single fitted-as-one Pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    estimator = _build_estimator(model_name, params, seed)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def suggest_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Per-estimator Optuna search space.

    Ranges are intentionally conservative so a 25-trial budget is sufficient
    on a CPU. Widen for a real production sweep.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    raise ValueError(f"Unsupported model {model_name!r}")
