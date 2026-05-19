"""Per-chunk scoring.

`score_chunk` is intentionally pure: it takes a chunk DataFrame and returns
both the predictions DataFrame and the chunk-level metrics. Pipeline-level
orchestration (writing, tracking, aggregating) lives in `pipeline.py` so
this module stays easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ChunkResult:
    """Outcome of scoring one chunk."""

    predictions: pd.DataFrame
    n_records: int
    n_positive: int
    proba_mean: float
    proba_std: float
    proba_p50: float
    proba_p95: float

    def to_metrics(self) -> dict:
        return {
            "n_records": int(self.n_records),
            "n_positive": int(self.n_positive),
            "positive_rate": float(self.n_positive / self.n_records) if self.n_records else 0.0,
            "proba_mean": float(self.proba_mean),
            "proba_std": float(self.proba_std),
            "proba_p50": float(self.proba_p50),
            "proba_p95": float(self.proba_p95),
        }


def _feature_frame(
    chunk: pd.DataFrame,
    drop_columns: Optional[List[str]],
    id_column: Optional[str],
) -> pd.DataFrame:
    """Return the frame to pass to `model.predict_proba`."""
    cols_to_drop: List[str] = []
    if drop_columns:
        cols_to_drop.extend(c for c in drop_columns if c in chunk.columns)
    if id_column and id_column in chunk.columns:
        cols_to_drop.append(id_column)
    return chunk.drop(columns=cols_to_drop) if cols_to_drop else chunk


def score_chunk(
    model: Any,
    chunk: pd.DataFrame,
    threshold: float,
    id_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
) -> ChunkResult:
    """Score one chunk and return predictions + summary stats.

    The output frame always has `prediction` and `probability` columns. If
    `id_column` is present in the input it is carried through as the first
    column so downstream systems can join predictions back to their entities.
    """
    if chunk.empty:
        empty = pd.DataFrame(columns=([id_column] if id_column else []) + ["probability", "prediction"])
        return ChunkResult(
            predictions=empty, n_records=0, n_positive=0,
            proba_mean=0.0, proba_std=0.0, proba_p50=0.0, proba_p95=0.0,
        )

    features = _feature_frame(chunk, drop_columns, id_column)
    proba_matrix = np.asarray(model.predict_proba(features))
    if proba_matrix.ndim != 2 or proba_matrix.shape[1] < 2:
        raise ValueError(
            "Model must implement predict_proba returning a (n, 2) array."
        )
    proba = proba_matrix[:, 1].astype(float)
    preds = (proba >= float(threshold)).astype(int)

    out = pd.DataFrame({"probability": proba, "prediction": preds}, index=chunk.index)
    if id_column and id_column in chunk.columns:
        out.insert(0, id_column, chunk[id_column].values)

    return ChunkResult(
        predictions=out.reset_index(drop=True),
        n_records=int(len(proba)),
        n_positive=int(preds.sum()),
        proba_mean=float(np.mean(proba)),
        proba_std=float(np.std(proba)),
        proba_p50=float(np.percentile(proba, 50)),
        proba_p95=float(np.percentile(proba, 95)),
    )


def aggregate_results(results: List[ChunkResult]) -> Tuple[dict, int]:
    """Combine per-chunk metrics into a single aggregate dict.

    `proba_*` aggregates are weighted by chunk size so the overall mean /
    std / percentile estimates respect the underlying record counts.
    """
    total = sum(r.n_records for r in results)
    if total == 0:
        return {
            "n_records": 0, "n_positive": 0, "positive_rate": 0.0,
            "proba_mean": 0.0, "proba_std": 0.0,
            "proba_p50": 0.0, "proba_p95": 0.0,
            "n_chunks": len(results),
        }, 0

    positives = sum(r.n_positive for r in results)
    weighted_mean = sum(r.proba_mean * r.n_records for r in results) / total
    # Pooled variance: weighted within-chunk variance + between-chunk variance.
    within = sum((r.proba_std ** 2) * r.n_records for r in results) / total
    between = sum(((r.proba_mean - weighted_mean) ** 2) * r.n_records for r in results) / total
    pooled_std = float(np.sqrt(within + between))

    weighted_p50 = sum(r.proba_p50 * r.n_records for r in results) / total
    weighted_p95 = sum(r.proba_p95 * r.n_records for r in results) / total

    return {
        "n_records": int(total),
        "n_positive": int(positives),
        "positive_rate": float(positives / total),
        "proba_mean": float(weighted_mean),
        "proba_std": pooled_std,
        "proba_p50": float(weighted_p50),
        "proba_p95": float(weighted_p95),
        "n_chunks": len(results),
    }, total
