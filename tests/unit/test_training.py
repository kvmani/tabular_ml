from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_ml_service.core.preprocessing import PreparationConfig
from tabular_ml_service.core.training import (
    TrainingOptions,
    predict_records,
    train_model,
)
from tabular_ml_service.errors import ApiError


def _classification_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    signal = rng.normal(size=rows)
    group = np.where(np.arange(rows) % 3 == 0, "lab", "plant")
    target = (signal + (group == "lab") * 0.8 > 0).astype(int)
    return pd.DataFrame(
        {
            "record_id": [f"R-{index:04}" for index in range(rows)],
            "signal": signal,
            "noise": rng.normal(scale=0.3, size=rows),
            "group": group,
            "target": target,
        }
    )


def _regression_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    feature = np.linspace(-3, 3, rows)
    group = np.where(np.arange(rows) % 2, "a", "b")
    target = 4.2 * feature + (group == "a") * 1.5 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame(
        {
            "row_id": [f"X-{index:04}" for index in range(rows)],
            "feature": feature,
            "group": group,
            "target": target,
        }
    )


def _options() -> TrainingOptions:
    return TrainingOptions(
        test_size=0.25,
        cv_folds=3,
        random_seed=42,
        permutation_importance=False,
    )


def test_classification_training_returns_metrics_diagnostics_and_inference() -> None:
    frame = _classification_frame()
    run = train_model(
        frame,
        target="target",
        algorithm_id="linear",
        parameters={"regularization": 1.0, "class_weight": "balanced"},
        preparation=PreparationConfig(),
        options=_options(),
    )
    assert run.task == "classification"
    assert run.result["metrics"]["balanced_accuracy"] >= 0.75
    assert len(run.result["cross_validation"]["metrics"]["accuracy"]["fold_scores"]) == 3
    assert run.result["diagnostics"]["confusion_matrix"]["labels"] == [0, 1]
    assert run.result["importance"]["native"]

    prediction = predict_records(
        run,
        frame.drop(columns=["target"]).head(2).to_dict(orient="records"),
    )
    assert len(prediction["rows"]) == 2
    assert set(prediction["rows"][0]["probabilities"]) == {"0", "1"}


def test_regression_training_returns_residuals_and_strong_fit() -> None:
    run = train_model(
        _regression_frame(),
        target="target",
        algorithm_id="linear",
        parameters={"regularization": 0.1, "class_weight": None},
        preparation=PreparationConfig(),
        options=_options(),
    )
    assert run.task == "regression"
    assert run.result["metrics"]["r2"] > 0.95
    assert run.result["metrics"]["rmse"] < 1.0
    assert run.result["diagnostics"]["residual_summary"]["std"] < 1.0


def test_permutation_importance_operates_on_original_features() -> None:
    run = train_model(
        _classification_frame(80),
        target="target",
        algorithm_id="decision_tree",
        parameters={"max_depth": 4, "min_samples_leaf": 2, "class_weight": None},
        preparation=PreparationConfig(),
        options=TrainingOptions(
            test_size=0.25,
            cv_folds=2,
            random_seed=2,
            permutation_importance=True,
        ),
    )
    names = {item["feature"] for item in run.result["importance"]["permutation"]}
    assert {"signal", "noise", "group"} <= names


def test_training_and_inference_validation_errors_are_actionable() -> None:
    tiny = _classification_frame(12)
    with pytest.raises(ApiError) as caught:
        train_model(
            tiny,
            target="target",
            algorithm_id="linear",
            parameters={},
            preparation=PreparationConfig(),
            options=_options(),
        )
    assert caught.value.code == "training.too_few_rows"

    run = train_model(
        _classification_frame(60),
        target="target",
        algorithm_id="linear",
        parameters={},
        preparation=PreparationConfig(),
        options=_options(),
    )
    with pytest.raises(ApiError) as missing:
        predict_records(run, [{"signal": 0.5}])
    assert missing.value.code == "inference.missing_columns"


@pytest.mark.parametrize(
    "payload",
    [
        {"test_size": 0.9},
        {"cv_folds": 1},
        {"random_seed": -1},
    ],
)
def test_training_options_validate_bounds(payload) -> None:
    with pytest.raises(ApiError):
        TrainingOptions.from_mapping(payload)
