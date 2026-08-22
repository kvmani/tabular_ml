from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_ml_service.core.visualization import visualization_payload
from tabular_ml_service.errors import ApiError


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(50, dtype=float),
            "y": np.arange(50, dtype=float) * 2 + 1,
            "group": ["a", "b"] * 25,
            "missing": [None] * 10 + list(range(40)),
        }
    )


def test_numeric_and_categorical_distributions() -> None:
    numeric = visualization_payload(
        _frame(),
        {"type": "distribution", "column": "x", "bins": 10, "kde": True},
    )
    categorical = visualization_payload(
        _frame(),
        {"type": "distribution", "column": "group"},
    )
    assert sum(numeric["data"]["counts"]) == 50
    assert len(numeric["data"]["kde"]["x"]) == 200
    assert categorical["data"]["labels"] == ["a", "b"]


def test_correlation_scatter_box_and_missingness() -> None:
    correlation = visualization_payload(
        _frame(),
        {"type": "correlation", "columns": ["x", "y"], "method": "spearman"},
    )
    scatter = visualization_payload(
        _frame(),
        {"type": "scatter", "x": "x", "y": "y", "color": "group"},
    )
    box = visualization_payload(
        _frame(),
        {"type": "box", "column": "y", "group": "group"},
    )
    missing = visualization_payload(_frame(), {"type": "missingness"})
    assert correlation["data"]["matrix"][0][1] == pytest.approx(1.0)
    assert scatter["data"]["trendline"]["r2"] == pytest.approx(1.0)
    assert len(box["data"]["groups"]) == 2
    assert missing["data"]["counts"][0] == 10


def test_visualization_validation_errors() -> None:
    with pytest.raises(ApiError) as invalid:
        visualization_payload(_frame(), {"type": "magic"})
    assert invalid.value.code == "visualization.invalid_type"
    with pytest.raises(ApiError, match="does not exist"):
        visualization_payload(
            _frame(),
            {"type": "distribution", "column": "not-real"},
        )
    with pytest.raises(ApiError, match="at least two"):
        visualization_payload(
            _frame(),
            {"type": "correlation", "columns": ["x"]},
        )
