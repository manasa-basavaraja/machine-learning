"""End-to-end orchestrator: load model, iterate chunks, score, write, log."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .loader import iter_input_chunks, load_model
from .scorer import ChunkResult, aggregate_results, score_chunk
from .tracking import Tracker, build_tracker, tracked_run
from .utils import file_sha256, get_logger, new_run_id, today_partition
from .writer import PartitionedParquetWriter


@dataclass
class RunSummary:
    """Returned by `run_batch_inference` so callers can assert on the run."""

    run_id: str
    run_date: str
    output_dir: Path
    n_records: int
    n_positive: int
    n_chunks: int
    n_files: int
    scoring_seconds: float
    metrics: Dict[str, float]
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "output_dir": str(self.output_dir),
            "n_records": self.n_records,
            "n_positive": self.n_positive,
            "n_chunks": self.n_chunks,
            "n_files": self.n_files,
            "scoring_seconds": self.scoring_seconds,
            "metrics": self.metrics,
            "params": self.params,
        }
        return d


def _build_params(config: Dict[str, Any], run_id: str, model_path: Path) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "input_path": str(config["input"]["path"]),
        "chunk_size": int(config["input"]["chunk_size"]),
        "id_column": config["input"].get("id_column"),
        "decision_threshold": float(config["prediction"]["threshold"]),
        "output_dir": str(config["output"]["dir"]),
    }


def _score_and_write_chunks(
    *,
    model: Any,
    config: Dict[str, Any],
    writer: PartitionedParquetWriter,
    tracker: Tracker,
    logger,
) -> List[ChunkResult]:
    threshold = float(config["prediction"]["threshold"])
    id_column = config["input"].get("id_column")
    drop_columns = list(config["input"].get("drop_columns") or [])
    chunk_size = int(config["input"]["chunk_size"])
    input_path = config["input"]["path"]

    results: List[ChunkResult] = []
    for idx, chunk in enumerate(iter_input_chunks(input_path, chunk_size)):
        result = score_chunk(
            model=model,
            chunk=chunk,
            threshold=threshold,
            id_column=id_column,
            drop_columns=drop_columns,
        )
        writer.write_chunk(result.predictions)
        per_chunk = result.to_metrics()
        tracker.log_metrics(per_chunk, step=idx)
        logger.info(
            "chunk=%d n=%d positives=%d mean_proba=%.4f",
            idx, per_chunk["n_records"], per_chunk["n_positive"],
            per_chunk["proba_mean"],
        )
        results.append(result)
    return results


def run_batch_inference(
    config: Dict[str, Any],
    tracker: Optional[Tracker] = None,
) -> RunSummary:
    """Run the full pipeline. Returns a RunSummary; never raises on bad rows
    inside chunks (loader/scorer issues still raise, by design).
    """
    logger = get_logger()

    model_path = Path(config["model"]["path"])
    model = load_model(model_path)
    logger.info("Loaded model from %s", model_path)

    run_id = new_run_id()
    run_date = today_partition()
    writer = PartitionedParquetWriter(
        output_dir=config["output"]["dir"],
        run_id=run_id,
        run_date=run_date,
    )
    tracker = tracker if tracker is not None else build_tracker(config)
    params = _build_params(config, run_id, model_path)

    run_name = f"{config.get('tracking', {}).get('run_name_prefix', 'batch')}_{run_id}"

    started = time.perf_counter()
    with tracked_run(tracker, run_name):
        tracker.log_params(params)
        results = _score_and_write_chunks(
            model=model, config=config, writer=writer, tracker=tracker, logger=logger,
        )
        elapsed = time.perf_counter() - started

        aggregate, total = aggregate_results(results)
        aggregate["scoring_seconds"] = float(elapsed)
        aggregate["records_per_second"] = float(total / elapsed) if elapsed > 0 else 0.0

        sample_size = int(config["output"].get("sample_size", 100) or 0)
        if sample_size and results:
            sample = pd.concat(
                [r.predictions.head(sample_size) for r in results if not r.predictions.empty],
                ignore_index=True,
            ).head(sample_size)
            sample_path = writer.write_sample(sample)
            tracker.log_artifact(sample_path)

        summary = {"params": params, "metrics": aggregate, "run_id": run_id, "run_date": run_date}
        summary_path = writer.write_summary(summary)
        writer.finalize()

        tracker.log_metrics(aggregate)
        tracker.log_artifact(summary_path)

    logger.info(
        "Run %s complete: %d records in %.2fs (%.1f rec/s)",
        run_id, aggregate["n_records"], elapsed, aggregate["records_per_second"],
    )

    return RunSummary(
        run_id=run_id,
        run_date=run_date,
        output_dir=writer.partition_dir,
        n_records=int(aggregate["n_records"]),
        n_positive=int(aggregate["n_positive"]),
        n_chunks=int(aggregate["n_chunks"]),
        n_files=int(writer.n_files),
        scoring_seconds=float(elapsed),
        metrics=aggregate,
        params=params,
    )
