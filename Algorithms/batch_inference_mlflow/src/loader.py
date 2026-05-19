"""Model loading and chunked input reading.

Two readers are exposed: one for CSV (via pandas' `chunksize`) and one for
Parquet (via PyArrow row groups). Both yield pandas DataFrames of roughly
`chunk_size` rows so the rest of the pipeline doesn't need to care about
the input format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import joblib
import pandas as pd


def load_model(path: str | Path) -> Any:
    """Load a joblib-serialized sklearn Pipeline (or any estimator)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found: {p.resolve()}")
    return joblib.load(p)


def _iter_csv(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        yield chunk


def _iter_parquet(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Iterate a parquet file (or directory of parts) one row group at a time,
    then rebatch to roughly `chunk_size` rows.
    """
    import pyarrow.parquet as pq  # local import keeps cold-start light

    dataset = pq.ParquetFile(str(path)) if path.is_file() else pq.ParquetDataset(str(path))
    buffer = []
    buffered_rows = 0

    def _flush():
        nonlocal buffer, buffered_rows
        if not buffer:
            return None
        df = pd.concat(buffer, ignore_index=True)
        buffer = []
        buffered_rows = 0
        return df

    if isinstance(dataset, pq.ParquetFile):
        for i in range(dataset.num_row_groups):
            df = dataset.read_row_group(i).to_pandas()
            buffer.append(df)
            buffered_rows += len(df)
            while buffered_rows >= chunk_size:
                merged = pd.concat(buffer, ignore_index=True)
                yield merged.iloc[:chunk_size]
                remainder = merged.iloc[chunk_size:]
                buffer = [remainder] if len(remainder) else []
                buffered_rows = len(remainder)
    else:
        table = dataset.read()
        df = table.to_pandas()
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size]

    final = _flush()
    if final is not None and len(final):
        yield final


def iter_input_chunks(path: str | Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Dispatch to the right reader based on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p.resolve()}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        yield from _iter_csv(p, chunk_size)
        return
    if suffix in {".parquet", ".pq"} or p.is_dir():
        yield from _iter_parquet(p, chunk_size)
        return
    raise ValueError(f"Unsupported input format: {p}")
