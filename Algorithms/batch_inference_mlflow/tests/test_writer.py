"""Tests for src.writer.PartitionedParquetWriter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.writer import PartitionedParquetWriter


@pytest.fixture()
def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "probability": [0.1, 0.7, 0.4],
            "prediction": [0, 1, 0],
        }
    )


def test_partition_layout_is_hive_style(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="run-A", run_date="2026-05-19",
    )
    writer.write_chunk(small_frame)
    expected = tmp_path / "run_date=2026-05-19" / "run_id=run-A"
    assert expected.is_dir()
    assert (expected / "part-00000.parquet").is_file()


def test_multiple_chunks_create_multiple_parts(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    writer.write_chunk(small_frame)
    writer.write_chunk(small_frame)
    writer.write_chunk(small_frame)
    files = sorted((tmp_path / "run_date=2026-05-19" / "run_id=r").glob("part-*.parquet"))
    assert [f.name for f in files] == ["part-00000.parquet", "part-00001.parquet", "part-00002.parquet"]
    assert writer.n_files == 3
    assert writer.n_rows == 3 * len(small_frame)


def test_empty_chunks_are_skipped(tmp_path):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    empty = pd.DataFrame(columns=["customer_id", "probability", "prediction"])
    assert writer.write_chunk(empty) is None
    assert writer.n_files == 0


def test_finalize_creates_success_marker(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    writer.write_chunk(small_frame)
    marker = writer.finalize()
    assert marker.is_file()
    assert marker.name == "_SUCCESS"


def test_writes_after_finalize_raise(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    writer.write_chunk(small_frame)
    writer.finalize()
    with pytest.raises(RuntimeError):
        writer.write_chunk(small_frame)


def test_summary_and_sample_are_written(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    sample_path = writer.write_sample(small_frame)
    summary_path = writer.write_summary({"n": 3, "mean": 0.4})
    assert sample_path.is_file() and sample_path.suffix == ".csv"
    assert summary_path.is_file() and summary_path.suffix == ".json"


def test_parquet_roundtrip_preserves_values(tmp_path, small_frame):
    writer = PartitionedParquetWriter(
        output_dir=tmp_path, run_id="r", run_date="2026-05-19",
    )
    writer.write_chunk(small_frame)
    written = pd.read_parquet(
        tmp_path / "run_date=2026-05-19" / "run_id=r" / "part-00000.parquet"
    )
    pd.testing.assert_frame_equal(
        written.sort_index(axis=1), small_frame.sort_index(axis=1),
        check_dtype=False,
    )
