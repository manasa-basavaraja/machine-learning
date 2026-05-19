"""CLI entry point.

CLI flags override the YAML config so a single config file can serve all
environments and operators tweak per-run knobs from the command line.
Exit codes: 0 on success, 1 on uncaught error, 2 on bad invocation.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .pipeline import run_batch_inference
from .utils import deep_merge, get_logger, load_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-inference",
        description="Run batch scoring with MLflow tracking.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model.path.")
    parser.add_argument("--input", default=None, help="Override input.path.")
    parser.add_argument("--output", default=None, help="Override output.dir.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override prediction.threshold.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override input.chunk_size.")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable MLflow tracking for this run.")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict:
    overrides: dict = {"model": {}, "input": {}, "output": {}, "prediction": {}, "tracking": {}}
    if args.model is not None:
        overrides["model"]["path"] = args.model
    if args.input is not None:
        overrides["input"]["path"] = args.input
    if args.chunk_size is not None:
        overrides["input"]["chunk_size"] = args.chunk_size
    if args.output is not None:
        overrides["output"]["dir"] = args.output
    if args.threshold is not None:
        overrides["prediction"]["threshold"] = args.threshold
    if args.no_tracking:
        overrides["tracking"]["enabled"] = False
    return {k: v for k, v in overrides.items() if v}


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger()

    try:
        config = load_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        return 2

    config = deep_merge(config, _overrides_from_args(args))

    try:
        summary = run_batch_inference(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Batch inference run failed")
        return 1

    logger.info(
        "OK run_id=%s records=%d files=%d output=%s",
        summary.run_id, summary.n_records, summary.n_files, summary.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
