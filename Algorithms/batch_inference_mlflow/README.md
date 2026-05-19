# Batch Inference Pipeline with MLflow Tracking

A production-shape batch scoring service. Loads a previously-trained
sklearn `Pipeline` artifact (for example, the one produced by
`customer_churn_pipeline`), streams a large input file through it in
chunks, writes Parquet predictions, and records the run to MLflow with
params, metrics, and artifacts so every scoring job is reproducible and
auditable.

## Why this shape

This is the scoring counterpart to a training pipeline, and it follows
the patterns that scale beyond a single notebook:

- **Chunked I/O.** Real input files don't fit in memory. The pipeline
  uses pandas' `chunksize` for CSV and PyArrow row groups for Parquet
  so the peak memory footprint is bounded by chunk size, not file
  size.
- **One Pipeline, two entry points.** The same fitted `Pipeline` from
  training is loaded here with `joblib`. No re-implementing
  preprocessing — eliminates training/serving skew by construction.
- **Tracking abstraction.** `tracking.Tracker` is the interface used by
  `pipeline.py`. `MLflowTracker` is the production implementation;
  `NoOpTracker` is the test/CI fallback so unit tests don't write to
  the MLflow store and the pipeline can run in environments without
  MLflow installed.
- **Partitioned output.** Predictions are written to
  `output_dir/run_date=YYYY-MM-DD/run_id=.../part-00000.parquet`, which
  matches the layout Athena / BigQuery / Spark expect for partition
  pruning.
- **Config-driven, CLI-overridable.** Every knob lives in
  `config/inference.yaml`; CLI flags override the config for ad-hoc runs.

## Project layout

```
batch_inference_mlflow/
├── config/
│   └── inference.yaml         # model + I/O + threshold + tracking settings
├── src/
│   ├── loader.py              # joblib model + chunked CSV/Parquet readers
│   ├── scorer.py              # score one chunk -> predictions + per-chunk metrics
│   ├── writer.py              # PartitionedParquetWriter (run_date / run_id)
│   ├── tracking.py            # Tracker interface + MLflow / NoOp impls
│   ├── pipeline.py            # orchestrator (load -> iter -> score -> write -> log)
│   ├── cli.py                 # CLI entry point with config overrides
│   └── utils.py               # config loader, logger, run-id helpers
├── tests/
│   ├── test_scorer.py
│   ├── test_writer.py
│   └── test_pipeline.py
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Score using the config defaults
python -m src.cli --config config/inference.yaml

# Override specific fields on the CLI
python -m src.cli \
    --config config/inference.yaml \
    --model ../customer_churn_pipeline/artifacts/model.joblib \
    --input data/new_customers.csv \
    --output artifacts/predictions \
    --threshold 0.45

# Run without MLflow (e.g. local debugging or environments without it)
python -m src.cli --config config/inference.yaml --no-tracking
```

After a run you'll have:

- `artifacts/predictions/run_date=YYYY-MM-DD/run_id=.../part-NNNNN.parquet`
- `artifacts/predictions/run_date=YYYY-MM-DD/run_id=.../_SUCCESS` marker
- `artifacts/predictions/run_date=YYYY-MM-DD/run_id=.../run_summary.json`
- An MLflow run under experiment `churn-batch-inference` (browse via
  `mlflow ui` against the same tracking URI).

## What gets tracked

**Params** (one set per run):
- `model_path`, `model_sha256`, `input_path`, `chunk_size`
- `decision_threshold`, `id_column`, `output_dir`

**Metrics** (final aggregate after all chunks score):
- `n_records`, `n_positive`, `positive_rate`
- `proba_mean`, `proba_std`, `proba_p50`, `proba_p95`
- `scoring_seconds`, `records_per_second`
- `n_chunks`

Per-chunk metrics are also logged with `step=chunk_idx` so progress is
visible in real time in the MLflow UI.

**Artifacts**:
- `run_summary.json` — full per-chunk + aggregate metrics
- `sample_predictions.csv` — first N predictions (configurable) for
  manual sanity-checking

## Tests

```bash
pytest tests/ -v
```

Tests cover the scorer (correct output schema, probability bounds, threshold
behavior), the writer (correct partition path layout, success marker,
multi-chunk concatenation), and the pipeline (end-to-end orchestration with
a fake estimator and the `NoOpTracker`).
