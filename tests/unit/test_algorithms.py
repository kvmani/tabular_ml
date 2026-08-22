from __future__ import annotations

import pytest

from tabular_ml_service.core.algorithms import (
    ALGORITHMS,
    algorithm_catalog,
    estimator_for,
)
from tabular_ml_service.errors import ApiError


def test_catalog_covers_major_classification_and_regression_families() -> None:
    classification = {item["id"] for item in algorithm_catalog("classification")}
    regression = {item["id"] for item in algorithm_catalog("regression")}
    required = {
        "linear",
        "knn",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "ada_boost",
        "svm",
        "sgd",
        "mlp",
    }
    assert required <= classification
    assert required <= regression
    assert {"naive_bayes"} <= classification
    assert {"lasso", "elastic_net"} <= regression


def test_every_supported_algorithm_builds_with_defaults() -> None:
    for spec in ALGORITHMS:
        for task in spec.tasks:
            returned, estimator, parameters = estimator_for(
                spec.id,
                task,
                {},
                random_seed=7,
            )
            assert returned is spec
            assert estimator is not None
            assert estimator.__class__.__module__.startswith("sklearn.")
            assert set(parameters) == {item.name for item in spec.parameters}


def test_parameter_validation_rejects_unknown_range_and_task() -> None:
    with pytest.raises(ApiError) as unknown:
        estimator_for("random_forest", "classification", {"magic": 1}, random_seed=1)
    assert unknown.value.code == "algorithm.unknown_parameters"

    with pytest.raises(ApiError) as invalid:
        estimator_for(
            "random_forest",
            "classification",
            {"n_estimators": 2},
            random_seed=1,
        )
    assert invalid.value.details["parameter"] == "n_estimators"

    with pytest.raises(ApiError, match="does not support"):
        estimator_for("naive_bayes", "regression", {}, random_seed=1)
