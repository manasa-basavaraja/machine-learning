"""Shared utilities: config loading, logging, run-id generation, file hashing."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config; raise descriptive errors on the usual mistakes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {p} must be a mapping at the top level.")
    return data


def get_logger(name: str = "bi", level: int = logging.INFO) -> logging.Logger:
    """Idempotent logger factory (safe to call repeatedly, e.g. from tests)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def new_run_id() -> str:
    """Compact, sortable run id: YYYYMMDDTHHMMSSZ_<8 hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def today_partition() -> str:
    """`run_date=YYYY-MM-DD` style partition key (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file; used to fingerprint the loaded model."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge used to apply CLI overrides onto the YAML config.

    Values in `override` win. Nested dicts are merged key-by-key; lists and
    scalars are replaced wholesale. `None` values in `override` are ignored
    so unset CLI flags don't clobber config defaults.
    """
    out = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
