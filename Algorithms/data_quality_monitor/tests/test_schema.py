"""Tests for src.schema.validate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema import validate


@pytest.fixture()
def schema() -> dict:
    return {
        "dataset": "toy",
        "row_count": {"min": 1},
        "columns": [
            {"name": "age", "dtype": "int", "nullable": False, "min": 0, "max": 120},
            {"name": "amount", "dtype": "float", "nullable": True, "min": 0.0},
            {
                "name": "status",
                "dtype": "str",
                "nullable": False,
                "allowed": ["active", "inactive", "pending"],
            },
        ],
    }


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 60],
            "amount": [12.5, np.nan, 99.9],
            "status": ["active", "inactive", "pending"],
        }
    )


def test_clean_frame_passes(schema):
    report = validate(_good_frame(), schema)
    assert report.passed
    assert report.issues == []


def test_null_in_non_nullable_fails(schema):
    df = _good_frame()
    df.loc[0, "age"] = np.nan
    report = validate(df, schema)
    assert not report.passed
    assert any(i.rule == "nullability" and i.column == "age" for i in report.issues)


def test_out_of_range_fails(schema):
    df = _good_frame()
    df.loc[0, "age"] = 200
    report = validate(df, schema)
    assert not report.passed
    rules = {(i.rule, i.column) for i in report.issues}
    assert ("range.max", "age") in rules


def test_unseen_category_fails(schema):
    df = _good_frame()
    df.loc[1, "status"] = "banned"
    report = validate(df, schema)
    assert not report.passed
    assert any(i.rule == "allowed_values" and i.column == "status" for i in report.issues)


def test_missing_column_fails(schema):
    df = _good_frame().drop(columns=["amount"])
    report = validate(df, schema)
    assert not report.passed
    assert any(i.rule == "column.missing" and i.column == "amount" for i in report.issues)


def test_extra_column_warns(schema):
    df = _good_frame()
    df["unexpected"] = [1, 2, 3]
    report = validate(df, schema)
    # Warnings don't fail the report.
    assert report.passed
    assert any(
        i.rule == "column.unexpected" and i.severity == "warning"
        for i in report.issues
    )


def test_dtype_mismatch_fails(schema):
    df = _good_frame()
    df["age"] = ["young", "old", "mid"]
    report = validate(df, schema)
    assert not report.passed
    assert any(i.rule == "dtype" and i.column == "age" for i in report.issues)
