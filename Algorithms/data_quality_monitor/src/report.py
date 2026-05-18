"""JSON + Markdown report renderers for schema and drift reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .drift import DriftReport
from .schema import SchemaReport


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _write_pair(out_base: Path, payload: Dict[str, Any], markdown: str) -> Dict[str, Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_schema_report(report: SchemaReport, out_base: Path) -> Dict[str, Path]:
    payload = {"generated_at": _utcnow_iso(), **report.to_dict()}

    header = (
        f"# Schema Validation Report\n\n"
        f"- **Dataset**: `{report.dataset}`\n"
        f"- **Generated**: `{payload['generated_at']}`\n"
        f"- **Rows**: `{report.n_rows}` | **Columns**: `{report.n_cols}`\n"
        f"- **Status**: {'PASSED' if report.passed else 'FAILED'}\n\n"
    )

    if not report.issues:
        body = "_All checks passed._\n"
    else:
        rows = [
            [
                i.severity.upper(),
                i.column or "-",
                i.rule,
                i.n_offending,
                i.message.replace("|", "\\|"),
            ]
            for i in report.issues
        ]
        body = "## Issues\n\n" + _md_table(
            ["Severity", "Column", "Rule", "# Offending", "Message"], rows
        )

    return _write_pair(out_base, payload, header + body)


def render_drift_report(report: DriftReport, out_base: Path) -> Dict[str, Path]:
    payload = {"generated_at": _utcnow_iso(), **report.to_dict()}

    header = (
        f"# Drift Report\n\n"
        f"- **Generated**: `{payload['generated_at']}`\n"
        f"- **Features checked**: `{len(report.features)}`\n"
        f"- **Warnings**: `{report.n_warnings}` | **Failures**: `{report.n_failures}`\n"
        f"- **Status**: {'PASSED' if report.passed else 'FAILED'}\n\n"
    )

    def _fmt(v):
        if v is None:
            return "-"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    rows = [
        [
            f.status.upper(),
            f.feature,
            f.kind,
            _fmt(f.psi),
            _fmt(f.ks),
            _fmt(f.chi_square),
            f.n_reference,
            f.n_current,
            "; ".join(f.notes) if f.notes else "-",
        ]
        for f in report.features
    ]
    body = "## Per-feature metrics\n\n" + _md_table(
        ["Status", "Feature", "Kind", "PSI", "KS", "Chi^2", "N ref", "N cur", "Notes"],
        rows,
    )

    return _write_pair(out_base, payload, header + body)
