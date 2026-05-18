"""Schema validation for tabular data.

The schema config is a list of column specs (name, dtype, nullable, optional
min / max for numerics, optional allowed values for categoricals). Validation
returns a structured report rather than raising mid-loop, so callers can
decide whether to abort, warn, or quarantine the bad rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


_NUMERIC_DTYPES = {"int", "float"}
_STRING_DTYPES = {"str", "string"}
_BOOL_DTYPES = {"bool"}
_ALL_DTYPES = _NUMERIC_DTYPES | _STRING_DTYPES | _BOOL_DTYPES


@dataclass
class SchemaIssue:
    """A single failed check, suitable for serialization."""

    column: Optional[str]
    rule: str
    severity: str           # "error" | "warning"
    message: str
    n_offending: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "n_offending": int(self.n_offending),
        }


@dataclass
class SchemaReport:
    dataset: str
    n_rows: int
    n_cols: int
    issues: List[SchemaIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
        }


def _check_dtype(series: pd.Series, expected: str) -> bool:
    """Return True if `series` is compatible with the declared dtype.

    We're deliberately permissive: e.g. a float column that happens to hold
    integer values is fine, and a column read from CSV as `object` of
    numeric-looking strings passes as numeric after coercion.
    """
    if expected in _NUMERIC_DTYPES:
        coerced = pd.to_numeric(series, errors="coerce")
        non_null_original = series.notna().sum()
        non_null_coerced = coerced.notna().sum()
        if non_null_coerced < non_null_original:
            return False
        if expected == "int":
            non_null = coerced.dropna()
            return bool(((non_null % 1) == 0).all())
        return True
    if expected in _STRING_DTYPES:
        return series.dropna().map(lambda v: isinstance(v, str)).all()
    if expected in _BOOL_DTYPES:
        return series.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all()
    raise ValueError(f"Unknown dtype {expected!r}; expected one of {_ALL_DTYPES}")


def _numeric_view(series: pd.Series) -> pd.Series:
    """Return a numeric-coerced view of a series for range checks."""
    return pd.to_numeric(series, errors="coerce")


def validate(df: pd.DataFrame, schema: Dict[str, Any]) -> SchemaReport:
    """Validate `df` against `schema` and return a SchemaReport."""
    dataset = str(schema.get("dataset", "unknown"))
    report = SchemaReport(dataset=dataset, n_rows=len(df), n_cols=df.shape[1])

    row_cfg = schema.get("row_count", {}) or {}
    min_rows = row_cfg.get("min")
    if min_rows is not None and len(df) < int(min_rows):
        report.issues.append(SchemaIssue(
            column=None,
            rule="row_count.min",
            severity="error",
            message=f"Dataset has {len(df)} rows, expected at least {min_rows}.",
        ))

    declared_cols = schema.get("columns", []) or []
    declared_names = {c["name"] for c in declared_cols}

    for name in declared_names - set(df.columns):
        report.issues.append(SchemaIssue(
            column=name,
            rule="column.missing",
            severity="error",
            message=f"Required column {name!r} is missing from the dataframe.",
        ))

    extra = set(df.columns) - declared_names
    for name in sorted(extra):
        report.issues.append(SchemaIssue(
            column=name,
            rule="column.unexpected",
            severity="warning",
            message=f"Column {name!r} is present in data but not declared in schema.",
        ))

    for spec in declared_cols:
        name = spec["name"]
        if name not in df.columns:
            continue
        col = df[name]
        expected_dtype = spec.get("dtype")

        if expected_dtype is not None and expected_dtype not in _ALL_DTYPES:
            report.issues.append(SchemaIssue(
                column=name,
                rule="dtype.unknown",
                severity="error",
                message=f"Schema declares unsupported dtype {expected_dtype!r}.",
            ))
            continue

        nullable = bool(spec.get("nullable", True))
        n_null = int(col.isna().sum())
        if not nullable and n_null > 0:
            report.issues.append(SchemaIssue(
                column=name,
                rule="nullability",
                severity="error",
                message=f"Column {name!r} declared non-nullable but has {n_null} nulls.",
                n_offending=n_null,
            ))

        if expected_dtype is not None and not _check_dtype(col, expected_dtype):
            report.issues.append(SchemaIssue(
                column=name,
                rule="dtype",
                severity="error",
                message=f"Column {name!r} is not compatible with dtype {expected_dtype!r}.",
            ))
            continue

        if expected_dtype in _NUMERIC_DTYPES:
            numeric = _numeric_view(col)
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None:
                below = int((numeric < float(lo)).sum())
                if below > 0:
                    report.issues.append(SchemaIssue(
                        column=name, rule="range.min", severity="error",
                        message=f"{below} value(s) below min={lo}.",
                        n_offending=below,
                    ))
            if hi is not None:
                above = int((numeric > float(hi)).sum())
                if above > 0:
                    report.issues.append(SchemaIssue(
                        column=name, rule="range.max", severity="error",
                        message=f"{above} value(s) above max={hi}.",
                        n_offending=above,
                    ))

        if expected_dtype in _STRING_DTYPES and "allowed" in spec:
            allowed = set(spec["allowed"])
            non_null = col.dropna()
            bad_mask = ~non_null.isin(allowed)
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                bad_examples = sorted(non_null[bad_mask].unique().tolist())[:5]
                report.issues.append(SchemaIssue(
                    column=name, rule="allowed_values", severity="error",
                    message=(
                        f"{n_bad} value(s) outside allowed set. "
                        f"Examples: {bad_examples}"
                    ),
                    n_offending=n_bad,
                ))

    return report
