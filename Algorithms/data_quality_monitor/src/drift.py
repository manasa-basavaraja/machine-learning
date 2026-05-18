"""Drift detection: PSI, KS, chi-square.

All metrics are computed in pure NumPy to avoid a SciPy dependency.
The conventions match what's commonly used in production monitoring:

- **PSI** uses fixed quantile bins derived from the *reference* sample so
  the same bin edges are reused for every future "current" snapshot.
- **KS** is the classical two-sample Kolmogorov-Smirnov statistic
  computed via merged-sort empirical CDFs.
- **Chi-square** is the Pearson statistic on a contingency table built
  from the union of categories seen in either sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureDrift:
    """Per-feature drift metrics + threshold decisions."""

    feature: str
    kind: str                       # "numeric" | "categorical"
    psi: float
    ks: Optional[float] = None
    chi_square: Optional[float] = None
    n_reference: int = 0
    n_current: int = 0
    status: str = "ok"              # "ok" | "warn" | "fail"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "psi": float(self.psi),
            "ks": None if self.ks is None else float(self.ks),
            "chi_square": None if self.chi_square is None else float(self.chi_square),
            "n_reference": int(self.n_reference),
            "n_current": int(self.n_current),
            "status": self.status,
            "notes": list(self.notes),
        }


@dataclass
class DriftReport:
    features: List[FeatureDrift] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.status == "fail" for f in self.features)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.features if f.status == "warn")

    @property
    def n_failures(self) -> int:
        return sum(1 for f in self.features if f.status == "fail")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "n_features": len(self.features),
            "n_warnings": self.n_warnings,
            "n_failures": self.n_failures,
            "features": [f.to_dict() for f in self.features],
        }


def _safe_drop_na(arr: pd.Series) -> np.ndarray:
    return arr.dropna().to_numpy()


def _quantile_bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Build monotonically increasing bin edges from reference quantiles.

    Duplicate edges (from heavy ties in the reference) are deduplicated and
    the outer edges are nudged to +/- inf so future samples can never fall
    outside the binning.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Proportion of `values` falling in each bin defined by `edges`."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / total


def psi_numeric(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index between two numeric samples."""
    if reference.size == 0 or current.size == 0:
        return float("nan")
    edges = _quantile_bin_edges(reference, n_bins)
    p_ref = _bin_proportions(reference, edges)
    p_cur = _bin_proportions(current, edges)
    p_ref = np.where(p_ref == 0, epsilon, p_ref)
    p_cur = np.where(p_cur == 0, epsilon, p_cur)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """PSI computed over the union of categories seen in either sample."""
    ref_counts = reference.value_counts(dropna=True)
    cur_counts = current.value_counts(dropna=True)
    categories = sorted(set(ref_counts.index) | set(cur_counts.index), key=str)

    ref_total = ref_counts.sum() or 1
    cur_total = cur_counts.sum() or 1

    p_ref = np.array(
        [ref_counts.get(c, 0) / ref_total for c in categories], dtype=float
    )
    p_cur = np.array(
        [cur_counts.get(c, 0) / cur_total for c in categories], dtype=float
    )
    p_ref = np.where(p_ref == 0, epsilon, p_ref)
    p_cur = np.where(p_cur == 0, epsilon, p_cur)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def ks_two_sample(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic computed via merged sort."""
    if reference.size == 0 or current.size == 0:
        return float("nan")
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    all_vals = np.concatenate([ref_sorted, cur_sorted])
    cdf_ref = np.searchsorted(ref_sorted, all_vals, side="right") / ref_sorted.size
    cdf_cur = np.searchsorted(cur_sorted, all_vals, side="right") / cur_sorted.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def chi_square_categorical(
    reference: pd.Series, current: pd.Series
) -> float:
    """Pearson chi-square statistic on the union-of-categories contingency table."""
    ref_counts = reference.value_counts(dropna=True)
    cur_counts = current.value_counts(dropna=True)
    categories = sorted(set(ref_counts.index) | set(cur_counts.index), key=str)

    observed = np.array(
        [[ref_counts.get(c, 0), cur_counts.get(c, 0)] for c in categories],
        dtype=float,
    )
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    if grand_total == 0:
        return float("nan")
    expected = row_totals @ col_totals / grand_total
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(contrib.sum())


def _classify(value: float, warn: Optional[float], fail: Optional[float]) -> str:
    if fail is not None and value >= float(fail):
        return "fail"
    if warn is not None and value >= float(warn):
        return "warn"
    return "ok"


def _merge_statuses(*statuses: str) -> str:
    """Worst-of merge: fail > warn > ok."""
    order = {"ok": 0, "warn": 1, "fail": 2}
    return max(statuses, key=lambda s: order.get(s, 0))


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: Dict[str, Any],
) -> DriftReport:
    """Run all configured drift checks and return a DriftReport."""
    binning = config.get("binning", {}) or {}
    n_bins = int(binning.get("numeric_bins", 10))
    epsilon = float(binning.get("smoothing_epsilon", 1e-4))

    features_cfg = config.get("features", {}) or {}
    report = DriftReport()

    for spec in features_cfg.get("numeric", []) or []:
        name = spec["name"]
        if name not in reference_df.columns or name not in current_df.columns:
            report.features.append(FeatureDrift(
                feature=name, kind="numeric", psi=float("nan"),
                status="fail", notes=["feature missing from reference or current"],
            ))
            continue
        ref = _safe_drop_na(pd.to_numeric(reference_df[name], errors="coerce"))
        cur = _safe_drop_na(pd.to_numeric(current_df[name], errors="coerce"))
        psi_val = psi_numeric(ref, cur, n_bins=n_bins, epsilon=epsilon)
        ks_val = ks_two_sample(ref, cur)
        psi_status = _classify(psi_val, spec.get("psi_warn"), spec.get("psi_fail"))
        ks_status = (
            "fail" if (spec.get("ks_fail") is not None and ks_val >= float(spec["ks_fail"]))
            else "ok"
        )
        report.features.append(FeatureDrift(
            feature=name, kind="numeric",
            psi=psi_val, ks=ks_val,
            n_reference=int(ref.size), n_current=int(cur.size),
            status=_merge_statuses(psi_status, ks_status),
        ))

    for spec in features_cfg.get("categorical", []) or []:
        name = spec["name"]
        if name not in reference_df.columns or name not in current_df.columns:
            report.features.append(FeatureDrift(
                feature=name, kind="categorical", psi=float("nan"),
                status="fail", notes=["feature missing from reference or current"],
            ))
            continue
        ref = reference_df[name].astype("object")
        cur = current_df[name].astype("object")
        psi_val = psi_categorical(ref, cur, epsilon=epsilon)
        chi_val = chi_square_categorical(ref, cur)
        new_cats = sorted(set(cur.dropna().unique()) - set(ref.dropna().unique()), key=str)
        notes: List[str] = []
        if new_cats:
            notes.append(f"new categories in current: {new_cats[:5]}")
        report.features.append(FeatureDrift(
            feature=name, kind="categorical",
            psi=psi_val, chi_square=chi_val,
            n_reference=int(ref.notna().sum()),
            n_current=int(cur.notna().sum()),
            status=_classify(psi_val, spec.get("psi_warn"), spec.get("psi_fail")),
            notes=notes,
        ))

    return report
