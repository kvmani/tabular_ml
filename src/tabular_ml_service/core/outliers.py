"""Outlier detection and reversible dataset treatment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from ..errors import ApiError

OutlierMethod = Literal["iqr", "zscore", "modified_zscore", "isolation_forest", "lof"]
OutlierAction = Literal["drop", "mask", "winsorize"]
OUTLIER_METHODS = {"iqr", "zscore", "modified_zscore", "isolation_forest", "lof"}
OUTLIER_ACTIONS = {"drop", "mask", "winsorize"}


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    method: OutlierMethod = "iqr"
    threshold: float = 1.5
    contamination: float = 0.05
    columns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> OutlierConfig:
        raw = payload or {}
        method = str(raw.get("method", "iqr"))
        if method not in OUTLIER_METHODS:
            raise ApiError(code="outliers.invalid", message=f"Unknown outlier method: {method}")
        threshold = float(raw.get("threshold", 1.5 if method == "iqr" else 3.0))
        if threshold <= 0:
            raise ApiError(code="outliers.invalid", message="Outlier threshold must be positive.")
        contamination = float(raw.get("contamination", 0.05))
        if not 0 < contamination <= 0.25:
            raise ApiError(
                code="outliers.invalid",
                message="Contamination must be greater than 0 and at most 0.25.",
            )
        columns = raw.get("columns", [])
        if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
            raise ApiError(
                code="outliers.invalid",
                message="Outlier columns must be a list of numeric column names.",
            )
        return cls(
            method=method,  # type: ignore[arg-type]
            threshold=threshold,
            contamination=contamination,
            columns=tuple(columns),
        )


@dataclass(slots=True)
class OutlierResult:
    config: OutlierConfig
    columns: list[str]
    row_mask: pd.Series
    cell_mask: pd.DataFrame
    scores: pd.Series
    column_counts: dict[str, int]

    def public_dict(self) -> dict[str, Any]:
        ranked = self.scores.sort_values(ascending=False)
        return {
            "method": self.config.method,
            "parameters": asdict(self.config),
            "columns": self.columns,
            "outlier_rows": int(self.row_mask.sum()),
            "outlier_ratio": float(self.row_mask.mean()),
            "column_counts": self.column_counts,
            "row_indices": [str(index) for index in self.row_mask[self.row_mask].index[:5_000]],
            "top_scores": [
                {"row": str(index), "score": float(score)}
                for index, score in ranked.head(100).items()
            ],
            "truncated": int(self.row_mask.sum()) > 5_000,
        }


def _numeric_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    selected = list(columns) if columns else frame.select_dtypes(include="number").columns.tolist()
    unknown = sorted(set(selected) - set(frame.columns))
    if unknown:
        raise ApiError(
            code="outliers.unknown_columns",
            message="Some outlier columns do not exist.",
            details={"columns": unknown},
        )
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in numeric if numeric[column].notna().sum() >= 4]
    if not usable:
        raise ApiError(
            code="outliers.no_numeric_columns",
            message="Choose at least one numeric column with four or more values.",
        )
    return cast(pd.DataFrame, numeric.loc[:, usable])


def _univariate_result(
    numeric: pd.DataFrame,
    config: OutlierConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    if config.method == "iqr":
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        spread = (q3 - q1).replace(0, np.nan)
        distance = (
            pd.concat(
                [
                    ((q1 - numeric) / spread).clip(lower=0),
                    ((numeric - q3) / spread).clip(lower=0),
                ],
                axis=0,
            )
            .groupby(level=0)
            .max()
        )
        cell_mask = distance > config.threshold
        scores = distance.max(axis=1).fillna(0)
    elif config.method == "zscore":
        deviation = ((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)).abs()
        cell_mask = deviation > config.threshold
        scores = deviation.max(axis=1).fillna(0)
    else:
        median = numeric.median()
        mad = (numeric - median).abs().median().replace(0, np.nan)
        deviation = (0.6745 * (numeric - median) / mad).abs()
        cell_mask = deviation > config.threshold
        scores = deviation.max(axis=1).fillna(0)
    return cell_mask.fillna(False), scores


def _multivariate_result(
    numeric: pd.DataFrame,
    config: OutlierConfig,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    values = SimpleImputer(strategy="median").fit_transform(numeric)
    values = RobustScaler().fit_transform(values)
    if config.method == "isolation_forest":
        detector: Any = IsolationForest(
            contamination=config.contamination,
            n_estimators=200,
            random_state=random_seed,
            n_jobs=1,
        )
        labels = detector.fit_predict(values)
        scores = -detector.score_samples(values)
    else:
        neighbours = max(2, min(20, len(numeric) - 1))
        detector = LocalOutlierFactor(
            n_neighbors=neighbours,
            contamination=config.contamination,
        )
        labels = detector.fit_predict(values)
        scores = -detector.negative_outlier_factor_
    row_mask = labels == -1
    cell_mask = pd.DataFrame(
        np.repeat(row_mask[:, None], len(numeric.columns), axis=1),
        index=numeric.index,
        columns=numeric.columns,
    )
    return cell_mask, pd.Series(scores, index=numeric.index)


def detect_outliers(
    frame: pd.DataFrame,
    config: OutlierConfig,
    *,
    random_seed: int = 42,
) -> OutlierResult:
    numeric = _numeric_frame(frame, config.columns)
    if config.method in {"iqr", "zscore", "modified_zscore"}:
        cell_mask, scores = _univariate_result(numeric, config)
    else:
        cell_mask, scores = _multivariate_result(numeric, config, random_seed)
    row_mask = cell_mask.any(axis=1)
    return OutlierResult(
        config=config,
        columns=list(numeric.columns),
        row_mask=row_mask,
        cell_mask=cell_mask,
        scores=scores,
        column_counts={str(column): int(cell_mask[column].sum()) for column in cell_mask.columns},
    )


def apply_outliers(
    frame: pd.DataFrame,
    result: OutlierResult,
    action: OutlierAction,
) -> pd.DataFrame:
    if action not in OUTLIER_ACTIONS:
        raise ApiError(code="outliers.invalid_action", message=f"Unknown outlier action: {action}")
    output = frame.copy(deep=True)
    if action == "drop":
        return output.loc[~result.row_mask].copy()
    if action == "mask":
        for column in result.columns:
            output.loc[result.cell_mask[column], column] = np.nan
        return output
    for column in result.columns:
        values = pd.to_numeric(output[column], errors="coerce")
        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
        output[column] = values.clip(lower=lower, upper=upper)
    return output
