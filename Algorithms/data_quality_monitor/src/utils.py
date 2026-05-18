"""Shared utilities: config loading and logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict, raising clear errors for the common mistakes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {p} must be a mapping at the top level.")
    return data


def get_logger(name: str = "dqm", level: int = logging.INFO) -> logging.Logger:
    """Idempotent logger factory; safe to call repeatedly (e.g. from tests)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
