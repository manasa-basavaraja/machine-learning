"""End-to-end test of the orchestrator with a fake model and NoOpTracker."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import run_batch_inference
from src.tracking import NoOpTracker


def _write_model(estimator, dst: Path) -> Path:
    joblib.dump(estimator, dst)
    return dst


def _write_csv(df: pd.DataFrame, dst: Path) -> Path:
    df.to_csv(dst, index=False)
    return dst


def test_run_batch_inference_writes_partitioned_output(tmp_path, fake_model):
    model_path = _write_model(fake_model, tmp_path / "model.joblib")
    df = pd.DataFrame(
        {
            "customer_id": list(range(25)),
            "tenure_months": list(range(25)),
            "monthly_charges": [50.0 + i for i in range(25)],
            "churn": [0] * 25,
        }
    )
    input_path = _write_csv(df, tmp_path / "input.csv")
    output_dir = tmp_path / "out"

    config = {
        "model": {"path": str(model_path)},
        "input": {
            "path": str(input_path),
            "chunk_size": 10,
            "id_column": "customer_id",
            "drop_columns": ["churn"],
        },
        "output": {"dir": str(output_dir), "sample_size": 5},
        "prediction": {"threshold": 0.5},
        "tracking": {"enabled": False},
    }

    summary = run_batch_inference(config, tracker=NoOpTracker())

    assert summary.n_records == 25
    assert summary.n_chunks == 3
    assert summary.n_files == 3
    assert summary.output_dir.is_dir()
    assert (summary.output_dir / "_SUCCESS").is_file()
    assert (summary.output_dir / "run_summary.json").is_file()
    assert (summary.output_dir / "sample_predictions.csv").is_file()

    part_files = sorted(summary.output_dir.glob("part-*.parquet"))
    assert len(part_files) == 3
    total_written = sum(len(pd.read_parquet(p)) for p in part_files)
    assert total_written == 25


def test_run_batch_inference_summary_json_matches_metrics(tmp_path, fake_model):
    model_path = _write_model(fake_model, tmp_path / "model.joblib")
    df = pd.DataFrame({"customer_id": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    input_path = _write_csv(df, tmp_path / "input.csv")
    output_dir = tmp_path / "out"

    config = {
        "model": {"path": str(model_path)},
        "input": {
            "path": str(input_path),
            "chunk_size": 2,
            "id_column": "customer_id",
            "drop_columns": [],
        },
        "output": {"dir": str(output_dir), "sample_size": 2},
        "prediction": {"threshold": 0.5},
        "tracking": {"enabled": False},
    }

    summary = run_batch_inference(config, tracker=NoOpTracker())
    summary_payload = json.loads(
        (summary.output_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    assert summary_payload["run_id"] == summary.run_id
    assert summary_payload["metrics"]["n_records"] == summary.n_records
    assert summary_payload["params"]["decision_threshold"] == 0.5
    assert summary_payload["params"]["model_sha256"]


def test_run_batch_inference_handles_threshold_zero(tmp_path, fake_model):
    model_path = _write_model(fake_model, tmp_path / "model.joblib")
    df = pd.DataFrame({"customer_id": [1, 2, 3], "x": [10, 20, 30]})
    input_path = _write_csv(df, tmp_path / "input.csv")
    output_dir = tmp_path / "out"

    config = {
        "model": {"path": str(model_path)},
        "input": {
            "path": str(input_path), "chunk_size": 3,
            "id_column": "customer_id", "drop_columns": [],
        },
        "output": {"dir": str(output_dir), "sample_size": 3},
        "prediction": {"threshold": 0.0},
        "tracking": {"enabled": False},
    }
    summary = run_batch_inference(config, tracker=NoOpTracker())
    assert summary.n_positive == summary.n_records
