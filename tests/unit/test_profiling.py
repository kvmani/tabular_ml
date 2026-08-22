from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_ml_service.core.profiling import profile_column, profile_dataset


def _mixed_frame() -> pd.DataFrame:
    rows = 36
    return pd.DataFrame(
        {
            "record_id": [f"R-{index:03}" for index in range(rows)],
            "age": [20 + index % 18 for index in range(rows)],
            "income": [35_000.0 + index * 1_125.5 for index in range(rows)],
            "joined_on": pd.date_range("2024-01-01", periods=rows, freq="D").astype(str),
            "department": ["lab", "plant", "office"] * 12,
            "notes": [
                f"Long free-form observation number {index} with context"
                for index in range(rows)
            ],
            "constant": "same",
            "promoted": [0, 1] * 18,
        }
    )


def test_profile_infers_semantic_types_and_roles() -> None:
    report = profile_dataset(_mixed_frame())
    columns = {item["name"]: item for item in report["columns"]}
    assert columns["record_id"]["semantic_type"] == "identifier"
    assert columns["record_id"]["suggested_role"] == "identifier"
    assert columns["income"]["semantic_type"] == "continuous"
    assert columns["joined_on"]["semantic_type"] == "datetime"
    assert columns["department"]["semantic_type"] == "categorical"
    assert columns["notes"]["semantic_type"] == "text"
    assert columns["constant"]["suggested_role"] == "ignore"
    assert report["target_candidates"][0]["column"] == "promoted"
    assert report["target_candidates"][0]["task"] == "classification"


def test_profile_reports_missing_duplicates_and_health() -> None:
    frame = _mixed_frame()
    frame.loc[:17, "income"] = np.nan
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = profile_dataset(frame)
    warning_codes = {item["code"] for item in report["warnings"]}
    expected = {"missing_values", "duplicate_rows", "constant_columns", "identifier_columns"}
    assert expected <= warning_codes
    assert report["overview"]["duplicate_rows"] == 1
    assert report["overview"]["health_score"] < 100


def test_profile_column_handles_empty_and_categorical_values() -> None:
    empty = profile_column("blank", pd.Series([None, None]))
    category = profile_column("color", pd.Series(["red", "blue", "red"]))
    assert empty.semantic_type == "empty"
    assert empty.suggested_role == "ignore"
    assert category.statistics["top_values"][0]["value"] == "red"
