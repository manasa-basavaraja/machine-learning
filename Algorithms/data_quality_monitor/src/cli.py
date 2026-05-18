"""CLI entry points.

Two commands are exposed:

    python -m src.cli validate-schema --schema ... --data ... --report ...
    python -m src.cli detect-drift  --config ... --reference ... --current ... --report ...

Exit codes are documented in the project README; the gist is 0 = clean,
1 = checks failed, 2 = bad invocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .drift import detect_drift
from .report import render_drift_report, render_schema_report
from .schema import validate
from .utils import get_logger, load_yaml


_EXIT_OK = 0
_EXIT_FAIL = 1
_EXIT_INVOCATION = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dqm", description="Data quality monitor.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_schema = sub.add_parser("validate-schema", help="Validate a CSV against a schema.")
    p_schema.add_argument("--schema", required=True)
    p_schema.add_argument("--data", required=True)
    p_schema.add_argument("--report", required=True, help="Path without extension.")

    p_drift = sub.add_parser("detect-drift", help="Compare reference vs current CSVs.")
    p_drift.add_argument("--config", required=True)
    p_drift.add_argument("--reference", required=True)
    p_drift.add_argument("--current", required=True)
    p_drift.add_argument("--report", required=True, help="Path without extension.")

    return parser


def _cmd_validate_schema(args: argparse.Namespace) -> int:
    logger = get_logger()
    schema = load_yaml(args.schema)
    data_path = Path(args.data)
    if not data_path.is_file():
        logger.error("Data file not found: %s", data_path)
        return _EXIT_INVOCATION

    df = pd.read_csv(data_path)
    report = validate(df, schema)
    paths = render_schema_report(report, Path(args.report))
    logger.info(
        "Schema validation %s. JSON=%s  MD=%s",
        "PASSED" if report.passed else "FAILED",
        paths["json"], paths["markdown"],
    )
    return _EXIT_OK if report.passed else _EXIT_FAIL


def _cmd_detect_drift(args: argparse.Namespace) -> int:
    logger = get_logger()
    config = load_yaml(args.config)

    for label, path_str in (("reference", args.reference), ("current", args.current)):
        if not Path(path_str).is_file():
            logger.error("%s file not found: %s", label, path_str)
            return _EXIT_INVOCATION

    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)
    report = detect_drift(ref_df, cur_df, config)
    paths = render_drift_report(report, Path(args.report))
    logger.info(
        "Drift detection %s. warnings=%d failures=%d JSON=%s MD=%s",
        "PASSED" if report.passed else "FAILED",
        report.n_warnings, report.n_failures,
        paths["json"], paths["markdown"],
    )
    return _EXIT_OK if report.passed else _EXIT_FAIL


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-schema":
        return _cmd_validate_schema(args)
    if args.command == "detect-drift":
        return _cmd_detect_drift(args)
    parser.print_help()
    return _EXIT_INVOCATION


if __name__ == "__main__":
    sys.exit(main())
