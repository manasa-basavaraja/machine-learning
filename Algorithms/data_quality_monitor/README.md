# Data Quality Monitor

A lightweight data-quality + drift-monitoring framework that ML pipelines
can plug in front of training and inference jobs. It covers the two
failure modes that cause the majority of silent ML regressions in
production:

1. **Schema breakage** — an upstream change renames a column, flips a
   type, introduces nulls in a non-nullable field, or starts emitting an
   unexpected categorical value. Catch this *before* training, not in a
   model post-mortem.
2. **Distribution drift** — the data still conforms to the schema but its
   distribution has shifted from what the model was trained on (different
   tenure mix, different payment-method mix, etc.). Catch this *before*
   the model starts making systematically worse predictions.

Designed to depend only on `numpy`, `pandas`, and `pyyaml` so it runs in
any production environment (lightweight Lambda containers, Airflow
workers, etc.) without dragging in heavy ML stacks.

## Why these implementations

- **Population Stability Index (PSI)** is the bank/risk industry standard
  for monitoring covariate shift on tabular features. It's bounded,
  symmetric, interpretable (`< 0.1` stable, `0.1–0.25` moderate shift,
  `> 0.25` significant shift), and works on both numeric and categorical
  data with the same formula.
- **KS statistic** (two-sample Kolmogorov–Smirnov) is the right tool for
  continuous numeric features when you want a single distribution-free
  scalar that's sensitive to shape changes, not just mean shifts.
- **Chi-square** for categorical features cross-checks PSI from a
  hypothesis-testing angle and surfaces new/missing categories.
- All three are implemented in pure NumPy so the package has no SciPy
  dependency, which keeps cold-start time and image size down.

## Project layout

```
data_quality_monitor/
├── config/
│   ├── schema.yaml           # data contract (columns, dtypes, ranges, allowed values)
│   └── drift.yaml            # baseline + current paths and per-feature drift thresholds
├── src/
│   ├── schema.py             # load schema + validate a DataFrame
│   ├── drift.py              # PSI / KS / chi-square implementations
│   ├── report.py             # render JSON + Markdown reports
│   ├── cli.py                # `validate-schema` and `detect-drift` entry points
│   └── utils.py              # config loader, logger
├── tests/
│   ├── test_schema.py
│   └── test_drift.py
├── reports/                  # generated at runtime
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Validate that a CSV conforms to the data contract
python -m src.cli validate-schema \
    --schema config/schema.yaml \
    --data path/to/data.csv \
    --report reports/schema_report

# 2) Detect drift between a reference (training) snapshot and current data
python -m src.cli detect-drift \
    --config config/drift.yaml \
    --reference path/to/reference.csv \
    --current   path/to/current.csv \
    --report reports/drift_report
```

Each command writes two files: `<report>.json` (machine-readable, suitable
for an Airflow XCom / monitoring dashboard) and `<report>.md` (human-
readable for PR review or Slack).

## Exit codes

Both CLI commands exit non-zero when issues are found, so they plug
straight into CI / Airflow:

| Exit code | Meaning                                  |
| --------- | ---------------------------------------- |
| `0`       | All checks passed                        |
| `1`       | At least one check failed (see report)   |
| `2`       | Bad invocation (missing file, bad YAML)  |

## Schema config

```yaml
dataset: telco_churn
row_count:
  min: 1
columns:
  - name: tenure_months
    dtype: int            # int | float | str | bool
    nullable: false
    min: 0
    max: 72
  - name: contract_type
    dtype: str
    nullable: false
    allowed: ["Month-to-month", "One year", "Two year"]
```

## Drift config

```yaml
features:
  numeric:
    - name: tenure_months
      psi_warn: 0.10
      psi_fail: 0.25
      ks_fail: 0.10
  categorical:
    - name: contract_type
      psi_warn: 0.10
      psi_fail: 0.25
binning:
  numeric_bins: 10          # quantile bins from the reference distribution
  smoothing_epsilon: 1.0e-4 # avoids log(0) in PSI
```

## Tests

```bash
pytest tests/ -v
```

Tests cover schema validation (good rows pass, bad dtypes / out-of-range
/ null violations / unseen categories fail) and drift math (identical
samples produce ~0 PSI, large mean shift produces large PSI, KS reacts
to shape changes, chi-square reacts to category proportion shifts).
