"""Shared utilities: config loading, logging, seeding."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty or not a mapping.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {cfg_path} must be a YAML mapping.")
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and the PYTHONHASHSEED env var for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to stdout and optionally a file.

    The logger is configured once per name; subsequent calls return the
    existing instance unchanged to avoid duplicate handlers when training
    scripts are re-imported (e.g. in tests).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Create the directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
