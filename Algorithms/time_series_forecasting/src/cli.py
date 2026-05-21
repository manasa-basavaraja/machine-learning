"""CLI entry point.

Usage:

    python -m src.cli --config config/forecast.yaml [--model NAME] [--horizon N]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .pipeline import run_pipeline
from .utils import get_logger, load_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsf",
        description="Time series forecasting with walk-forward backtesting.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None,
                        help="Override model.name (naive | seasonal_naive | moving_average | ridge).")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override forecast.horizon.")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = get_logger()

    try:
        config = load_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        return 2

    if args.model is not None:
        config.setdefault("model", {})["name"] = args.model
    if args.horizon is not None:
        config.setdefault("forecast", {})["horizon"] = int(args.horizon)

    try:
        artifacts = run_pipeline(config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Pipeline run failed")
        return 1

    logger.info(
        "OK n_folds=%d horizon=%d aggregate=%s",
        len(artifacts.backtest.folds),
        artifacts.backtest.horizon,
        {k: round(v, 4) for k, v in artifacts.backtest.aggregate.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
