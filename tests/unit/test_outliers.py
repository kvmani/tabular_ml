from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_ml_service.core.outliers import (
    OutlierConfig,
    apply_outliers,
    detect_outliers,
)
from tabular_ml_service.errors import ApiError


def _frame() -> pd.DataFrame:
    values = [float(index) for index in range(30)] + [500.0]
    return pd.DataFrame(
        {
            "measurement": values,
            "secondary": [value * 0.5 for value in values],
            "group": ["a"] * len(values),
        }
    )


@pytest.mark.parametrize("method", ["iqr", "zscore", "modified_zscore"])
def test_univariate_detectors_find_extreme_row(method: str) -> None:
    result = detect_outliers(
        _frame(),
        OutlierConfig.from_mapping({"method": method, "threshold": 1.5}),
    )
    assert result.row_mask.iloc[-1]
    assert result.public_dict()["outlier_rows"] >= 1
    assert result.column_counts["measurement"] >= 1


@pytest.mark.parametrize("method", ["isolation_forest", "lof"])
def test_multivariate_detectors_return_ranked_scores(method: str) -> None:
    result = detect_outliers(
        _frame(),
        OutlierConfig.from_mapping({"method": method, "contamination": 0.1}),
    )
    report = result.public_dict()
    assert report["outlier_rows"] >= 1
    assert report["top_scores"][0]["score"] >= report["top_scores"][-1]["score"]


def test_drop_mask_and_winsorize_are_non_mutating() -> None:
    frame = _frame()
    result = detect_outliers(frame, OutlierConfig())
    dropped = apply_outliers(frame, result, "drop")
    masked = apply_outliers(frame, result, "mask")
    winsorized = apply_outliers(frame, result, "winsorize")
    assert len(dropped) < len(frame)
    assert masked["measurement"].isna().sum() >= 1
    assert winsorized["measurement"].max() < frame["measurement"].max()
    assert frame["measurement"].max() == 500.0


def test_outlier_validation_is_actionable() -> None:
    with pytest.raises(ApiError, match="Unknown"):
        OutlierConfig.from_mapping({"method": "magic"})
    with pytest.raises(ApiError, match="numeric"):
        detect_outliers(_frame()[["group"]], OutlierConfig())
    with pytest.raises(ApiError, match="Unknown outlier action"):
        apply_outliers(_frame(), detect_outliers(_frame(), OutlierConfig()), "erase")


def test_missing_numeric_values_are_supported() -> None:
    frame = _frame()
    frame.loc[2:5, "measurement"] = np.nan
    result = detect_outliers(
        frame,
        OutlierConfig.from_mapping({"method": "isolation_forest"}),
    )
    assert len(result.row_mask) == len(frame)

