"""Tests for src.scorer."""

from __future__ import annotations

import pandas as pd

from src.scorer import aggregate_results, score_chunk


def test_score_chunk_returns_required_columns(fake_model, toy_chunk):
    result = score_chunk(fake_model, toy_chunk, threshold=0.5, id_column="customer_id")
    assert list(result.predictions.columns) == ["customer_id", "probability", "prediction"]
    assert len(result.predictions) == len(toy_chunk)


def test_score_chunk_probabilities_in_unit_interval(fake_model, toy_chunk):
    result = score_chunk(fake_model, toy_chunk, threshold=0.5, id_column="customer_id")
    proba = result.predictions["probability"]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_threshold_changes_positive_count(fake_model, toy_chunk):
    low = score_chunk(fake_model, toy_chunk, threshold=0.1, id_column="customer_id")
    high = score_chunk(fake_model, toy_chunk, threshold=0.9, id_column="customer_id")
    assert low.n_positive >= high.n_positive


def test_drop_columns_are_removed_before_predict(fake_model, toy_chunk):
    chunk = toy_chunk.copy()
    chunk["churn"] = [0, 1, 0, 1, 0]
    result = score_chunk(
        fake_model, chunk, threshold=0.5,
        id_column="customer_id", drop_columns=["churn"],
    )
    assert len(result.predictions) == len(chunk)
    assert "churn" not in result.predictions.columns


def test_id_column_carried_through_in_order(fake_model, toy_chunk):
    result = score_chunk(fake_model, toy_chunk, threshold=0.5, id_column="customer_id")
    assert result.predictions["customer_id"].tolist() == toy_chunk["customer_id"].tolist()


def test_score_chunk_empty_returns_empty_frame(fake_model):
    empty = pd.DataFrame({"customer_id": [], "tenure_months": []})
    result = score_chunk(fake_model, empty, threshold=0.5, id_column="customer_id")
    assert result.n_records == 0
    assert result.predictions.empty


def test_aggregate_results_sums_counts(fake_model, toy_chunk):
    r1 = score_chunk(fake_model, toy_chunk, threshold=0.5, id_column="customer_id")
    r2 = score_chunk(fake_model, toy_chunk, threshold=0.5, id_column="customer_id")
    agg, total = aggregate_results([r1, r2])
    assert total == r1.n_records + r2.n_records
    assert agg["n_records"] == total
    assert agg["n_positive"] == r1.n_positive + r2.n_positive
    assert agg["n_chunks"] == 2


def test_aggregate_results_handles_empty_list():
    agg, total = aggregate_results([])
    assert total == 0
    assert agg["n_records"] == 0
    assert agg["positive_rate"] == 0.0
