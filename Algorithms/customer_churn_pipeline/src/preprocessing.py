"""Preprocessing factory.

The single source of truth for feature transformations. Both training and
inference go through the `ColumnTransformer` returned here, which keeps the
fitted state (imputer statistics, one-hot vocabularies, scaler means)
attached to the model artifact.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_ohe() -> OneHotEncoder:
    """OneHotEncoder constructor that is portable across sklearn versions.

    `sparse_output` replaced `sparse` in sklearn 1.2. We try the new kwarg
    first and fall back to the old one so the project runs on both.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build the ColumnTransformer used for both training and inference.

    Numeric columns: median impute then standard-scale.
    Categorical columns: most-frequent impute then one-hot encode with
    `handle_unknown="ignore"` so unseen categories at inference time
    become an all-zero row instead of raising.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def feature_lists_from_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Pull the numeric/categorical feature lists out of the config."""
    feats = config.get("features", {})
    return {
        "numeric": list(feats.get("numeric", [])),
        "categorical": list(feats.get("categorical", [])),
    }
