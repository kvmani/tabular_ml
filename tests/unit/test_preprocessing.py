from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_ml_service.core.preprocessing import (
    DatePartsTransformer,
    PreparationConfig,
    build_preprocessor,
    feature_plan,
)
from tabular_ml_service.errors import ApiError


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"R-{index:03}" for index in range(30)],
            "amount": [float(index) if index != 4 else np.nan for index in range(30)],
            "count": [index % 5 for index in range(30)],
            "group": ["alpha", "beta", None] * 10,
            "observed": pd.date_range("2025-01-01", periods=30, freq="D").astype(str),
            "target": [0, 1] * 15,
        }
    )


def test_feature_plan_groups_columns_and_detects_task() -> None:
    plan = feature_plan(_frame(), "target", PreparationConfig())
    assert plan["task"] == "classification"
    assert set(plan["groups"]["numeric"]) == {"amount", "count"}
    assert plan["groups"]["categorical"] == ["group"]
    assert plan["groups"]["datetime"] == ["observed"]
    assert {"record_id", "target"} <= set(plan["groups"]["excluded"])
    assert plan["estimated_output_features"] == 9


def test_preprocessor_fits_missing_mixed_data_without_leakage_target() -> None:
    frame = _frame()
    transformer, plan = build_preprocessor(frame, "target", PreparationConfig())
    transformed = transformer.fit_transform(frame.drop(columns=["target"]))
    names = transformer.get_feature_names_out().tolist()
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] == len(names)
    assert "target" not in names
    assert "observed_year" in names
    assert plan["config"]["numeric_imputer"] == "median"


def test_knn_ordinal_and_date_drop_variants() -> None:
    config = PreparationConfig.from_mapping(
        {
            "numeric_imputer": "knn",
            "encoding": "ordinal",
            "scaling": "robust",
            "date_strategy": "drop",
            "rare_category_threshold": 0.05,
        }
    )
    transformer, plan = build_preprocessor(_frame(), "target", config)
    transformed = transformer.fit_transform(_frame().drop(columns=["target"]))
    assert transformed.shape[0] == 30
    assert plan["groups"]["datetime"] == []
    assert "observed" in plan["groups"]["excluded"]


def test_date_parts_handles_missing_values() -> None:
    values = pd.DataFrame({"when": ["2024-02-29", None]})
    transformed = DatePartsTransformer().fit_transform(values)
    assert transformed.shape == (2, 5)
    assert transformed[0].tolist()[:3] == [2024.0, 2.0, 29.0]
    assert np.isnan(transformed[1]).all()


@pytest.mark.parametrize(
    "payload",
    [
        {"numeric_imputer": "magic"},
        {"encoding": "hash"},
        {"rare_category_threshold": 0.5},
        {"exclude_columns": "not-a-list"},
    ],
)
def test_preparation_config_rejects_invalid_values(payload) -> None:
    with pytest.raises(ApiError):
        PreparationConfig.from_mapping(payload)


def test_feature_plan_rejects_unknown_target_and_no_features() -> None:
    with pytest.raises(ApiError, match="does not exist"):
        feature_plan(_frame(), "missing", PreparationConfig())
    with pytest.raises(ApiError, match="No usable"):
        feature_plan(
            _frame()[["record_id", "target"]],
            "target",
            PreparationConfig(),
        )
