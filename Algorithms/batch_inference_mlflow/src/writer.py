"""Partitioned Parquet writer.

Output layout follows the Hive-style partition convention that Athena,
BigQuery, Spark, and DuckDB all recognize, so downstream consumers can
prune partitions instead of scanning every file:

    <output_dir>/
        run_date=YYYY-MM-DD/
            run_id=<run_id>/
                part-00000.parquet
                part-00001.parquet
                ...
                _SUCCESS
                run_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class PartitionedParquetWriter:
    """Streaming writer: append chunks, then `finalize` to close the run."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        run_date: str,
        compression: str = "snappy",
    ):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.run_date = run_date
        self.compression = compression

        self.partition_dir = (
            self.output_dir
            / f"run_date={run_date}"
            / f"run_id={run_id}"
        )
        self.partition_dir.mkdir(parents=True, exist_ok=True)

        self._chunk_idx = 0
        self._total_rows = 0
        self._finalized = False

    @property
    def n_files(self) -> int:
        return self._chunk_idx

    @property
    def n_rows(self) -> int:
        return self._total_rows

    def write_chunk(self, df: pd.DataFrame) -> Optional[Path]:
        """Write `df` as the next part file. Empty frames are skipped."""
        if self._finalized:
            raise RuntimeError("Writer has been finalized; cannot write more chunks.")
        if df is None or df.empty:
            return None
        part_path = self.partition_dir / f"part-{self._chunk_idx:05d}.parquet"
        df.to_parquet(part_path, index=False, compression=self.compression)
        self._chunk_idx += 1
        self._total_rows += len(df)
        return part_path

    def write_sample(self, df: pd.DataFrame, filename: str = "sample_predictions.csv") -> Path:
        """Write a sample (head) of the predictions for manual sanity-checks."""
        path = self.partition_dir / filename
        df.to_csv(path, index=False)
        return path

    def write_summary(self, summary: Dict[str, Any], filename: str = "run_summary.json") -> Path:
        path = self.partition_dir / filename
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def finalize(self) -> Path:
        """Drop a `_SUCCESS` marker so consumers know the run is complete."""
        marker = self.partition_dir / "_SUCCESS"
        marker.write_text("", encoding="utf-8")
        self._finalized = True
        return marker
