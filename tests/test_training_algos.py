from __future__ import annotations

import pytest

from backend.app.services.data_manager import data_manager
from backend.app.services.model_training import model_trainer

TORCH_AVAILABLE = model_trainer.torch_available


NON_TORCH_CASES = [
    ("titanic", "Survived", "classification", "logistic_regression", None),
    (
        "synthetic_sales_forecast",
        "quarterly_revenue_k",
        "regression",
        "random_forest",
        None,
    ),
]

TORCH_CASES = [
    (
        "titanic",
        "Survived",
        "classification",
        "neural_network",
        {"epochs": 2, "batch_size": 16, "hidden_layers": [32, 16]},
    ),
]

TRAIN_CASES = NON_TORCH_CASES + (TORCH_CASES if TORCH_AVAILABLE else [])


@pytest.mark.parametrize(
    "dataset_key,target,task,algorithm,hyperparameters",
    TRAIN_CASES,
)
def test_training_runs(dataset_key, target, task, algorithm, hyperparameters):
    metadata = data_manager.load_sample_dataset(dataset_key)
    df = data_manager.get_dataset(metadata.dataset_id)
    result = model_trainer.train(
        dataset_id=metadata.dataset_id,
        df=df,
        target_column=target,
        task_type=task,
        algorithm=algorithm,
        hyperparameters=dict(hyperparameters) if hyperparameters else None,
    )

    assert "validation" in result.artifact.metrics
    assert "test" in result.artifact.metrics
    assert result.artifact.metrics["validation"]
    assert result.artifact.metrics["test"]


def test_algorithm_catalog_matches_torch_availability():
    keys = {algo["key"] for algo in model_trainer.available_algorithms()}
    if TORCH_AVAILABLE:
        assert "neural_network" in keys
    else:
        assert "neural_network" not in keys
