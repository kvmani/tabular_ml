"""Leakage-safe preprocessing configuration and sklearn transformer building."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from ..errors import ApiError
from .profiling import ColumnProfile, profile_column

NumericImputer = Literal["mean", "median", "most_frequent", "knn"]
CategoricalImputer = Literal["most_frequent", "constant"]
ScalerName = Literal["none", "standard", "minmax", "robust"]
EncoderName = Literal["onehot", "ordinal"]
DateStrategy = Literal["components", "drop"]

NUMERIC_IMPUTERS = {"mean", "median", "most_frequent", "knn"}
CATEGORICAL_IMPUTERS = {"most_frequent", "constant"}
SCALERS = {"none", "standard", "minmax", "robust"}
ENCODERS = {"onehot", "ordinal"}
DATE_STRATEGIES = {"components", "drop"}


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    numeric_imputer: NumericImputer = "median"
    categorical_imputer: CategoricalImputer = "most_frequent"
    scaling: ScalerName = "standard"
    encoding: EncoderName = "onehot"
    add_missing_indicators: bool = True
    date_strategy: DateStrategy = "components"
    rare_category_threshold: float = 0.01
    exclude_columns: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> PreparationConfig:
        raw = payload or {}
        numeric = str(raw.get("numeric_imputer", "median"))
        categorical = str(raw.get("categorical_imputer", "most_frequent"))
        scaling = str(raw.get("scaling", "standard"))
        encoding = str(raw.get("encoding", "onehot"))
        date_strategy = str(raw.get("date_strategy", "components"))
        if numeric not in NUMERIC_IMPUTERS:
            raise ApiError(
                code="preparation.invalid",
                message=f"Unknown numeric imputer: {numeric}",
            )
        if categorical not in CATEGORICAL_IMPUTERS:
            raise ApiError(
                code="preparation.invalid",
                message=f"Unknown categorical imputer: {categorical}",
            )
        if scaling not in SCALERS:
            raise ApiError(code="preparation.invalid", message=f"Unknown scaling method: {scaling}")
        if encoding not in ENCODERS:
            raise ApiError(
                code="preparation.invalid",
                message=f"Unknown encoding method: {encoding}",
            )
        if date_strategy not in DATE_STRATEGIES:
            raise ApiError(
                code="preparation.invalid",
                message=f"Unknown date strategy: {date_strategy}",
            )
        threshold = float(raw.get("rare_category_threshold", 0.01))
        if not 0 <= threshold <= 0.2:
            raise ApiError(
                code="preparation.invalid",
                message="Rare-category threshold must be between 0 and 0.2.",
            )
        excluded_raw = raw.get("exclude_columns", [])
        if not isinstance(excluded_raw, list) or not all(
            isinstance(item, str) for item in excluded_raw
        ):
            raise ApiError(
                code="preparation.invalid",
                message="exclude_columns must be a list of column names.",
            )
        return cls(
            numeric_imputer=numeric,  # type: ignore[arg-type]
            categorical_imputer=categorical,  # type: ignore[arg-type]
            scaling=scaling,  # type: ignore[arg-type]
            encoding=encoding,  # type: ignore[arg-type]
            add_missing_indicators=bool(raw.get("add_missing_indicators", True)),
            date_strategy=date_strategy,  # type: ignore[arg-type]
            rare_category_threshold=threshold,
            exclude_columns=tuple(excluded_raw),
        )


class DatePartsTransformer(TransformerMixin, BaseEstimator):
    """Expand one or more date columns into stable numeric calendar features."""

    def fit(self, values: Any, y: Any = None) -> DatePartsTransformer:
        self.n_features_in_ = np.asarray(values).shape[1]
        return self

    def transform(self, values: Any) -> np.ndarray:
        frame = pd.DataFrame(values)
        output: list[np.ndarray] = []
        for column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            output.extend(
                [
                    parsed.dt.year.to_numpy(dtype=float),
                    parsed.dt.month.to_numpy(dtype=float),
                    parsed.dt.day.to_numpy(dtype=float),
                    parsed.dt.dayofweek.to_numpy(dtype=float),
                    parsed.dt.dayofyear.to_numpy(dtype=float),
                ]
            )
        return np.column_stack(output)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        features = (
            [str(item) for item in input_features]
            if input_features is not None
            else [f"date_{index}" for index in range(self.n_features_in_)]
        )
        suffixes = ("year", "month", "day", "dayofweek", "dayofyear")
        return np.asarray([f"{feature}_{suffix}" for feature in features for suffix in suffixes])


def feature_plan(
    frame: pd.DataFrame,
    target: str,
    config: PreparationConfig,
) -> dict[str, Any]:
    if target not in frame.columns:
        raise ApiError(code="target.not_found", message=f"Target column '{target}' does not exist.")
    unknown = sorted(set(config.exclude_columns) - set(frame.columns))
    if unknown:
        raise ApiError(
            code="preparation.unknown_columns",
            message="Some excluded columns do not exist.",
            details={"columns": unknown},
        )
    groups: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "excluded": [],
    }
    column_profiles: dict[str, ColumnProfile] = {}
    for column in frame.columns:
        name = str(column)
        profile = profile_column(name, frame[column])
        column_profiles[name] = profile
        if name == target or name in config.exclude_columns:
            groups["excluded"].append(name)
        elif profile.semantic_type in {"integer", "continuous"}:
            groups["numeric"].append(name)
        elif profile.semantic_type in {"boolean", "categorical"}:
            groups["categorical"].append(name)
        elif profile.semantic_type == "datetime" and config.date_strategy == "components":
            groups["datetime"].append(name)
        else:
            groups["excluded"].append(name)
    feature_count = sum(len(groups[name]) for name in ("numeric", "categorical", "datetime"))
    if feature_count == 0:
        raise ApiError(
            code="preparation.no_features",
            message="No usable feature columns remain after applying the current exclusions.",
        )
    target_profile = column_profiles[target]
    classification_limit = max(12, min(50, int(len(frame) ** 0.5) + 2))
    task = (
        "classification"
        if target_profile.semantic_type in {"boolean", "categorical"}
        or (
            target_profile.semantic_type == "integer"
            and target_profile.unique <= classification_limit
        )
        else "regression"
    )
    warnings: list[dict[str, str]] = []
    if target_profile.missing:
        warnings.append(
            {
                "code": "target_missing",
                "message": (
                    f"{target_profile.missing} rows with missing target values "
                    "will be excluded before splitting."
                ),
            }
        )
    if target_profile.semantic_type in {"identifier", "text", "empty", "constant", "datetime"}:
        warnings.append(
            {
                "code": "unusual_target",
                "message": (
                    f"The target is inferred as {target_profile.semantic_type}; "
                    "confirm that it represents the outcome."
                ),
            }
        )
    categorical_levels = sum(column_profiles[name].unique for name in groups["categorical"])
    estimated_features = (
        len(groups["numeric"])
        + len(groups["datetime"]) * 5
        + (
            categorical_levels
            if config.encoding == "onehot"
            else len(groups["categorical"])
        )
    )
    if estimated_features > 1_000:
        warnings.append(
            {
                "code": "high_dimensional_encoding",
                "message": (
                    f"One-hot encoding may produce about {estimated_features:,} features. "
                    "Raise the rare-category threshold or exclude high-cardinality columns."
                ),
            }
        )
    return {
        "target": target,
        "task": task,
        "groups": groups,
        "estimated_output_features": estimated_features,
        "target_missing_rows": target_profile.missing,
        "usable_rows": target_profile.non_missing,
        "config": asdict(config),
        "warnings": warnings,
    }


def _numeric_pipeline(config: PreparationConfig) -> Pipeline:
    if config.numeric_imputer == "knn":
        imputer: Any = KNNImputer(
            n_neighbors=5,
            weights="distance",
            add_indicator=config.add_missing_indicators,
        )
    else:
        imputer = SimpleImputer(
            strategy=config.numeric_imputer,
            add_indicator=config.add_missing_indicators,
        )
    steps: list[tuple[str, Any]] = [("imputer", imputer)]
    scaler: Any | None = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": None,
    }[config.scaling]
    if scaler is not None:
        steps.append(("scaler", scaler))
    return Pipeline(steps)


def _categorical_pipeline(config: PreparationConfig) -> Pipeline:
    fill_value = "__missing__" if config.categorical_imputer == "constant" else None
    imputer = SimpleImputer(
        strategy=config.categorical_imputer,
        fill_value=fill_value,
        add_indicator=config.add_missing_indicators,
    )
    if config.encoding == "onehot":
        encoder: Any = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=config.rare_category_threshold or None,
            sparse_output=True,
        )
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    return Pipeline([("imputer", imputer), ("encoder", encoder)])


def build_preprocessor(
    frame: pd.DataFrame,
    target: str,
    config: PreparationConfig,
) -> tuple[ColumnTransformer, dict[str, Any]]:
    """Create an unfitted transformer. Fit only inside validation/training folds."""
    plan = feature_plan(frame, target, config)
    groups = plan["groups"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if groups["numeric"]:
        transformers.append(("numeric", _numeric_pipeline(config), groups["numeric"]))
    if groups["categorical"]:
        transformers.append(("categorical", _categorical_pipeline(config), groups["categorical"]))
    if groups["datetime"]:
        date_pipeline = Pipeline(
            [
                ("parts", DatePartsTransformer()),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=config.add_missing_indicators,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("datetime", date_pipeline, groups["datetime"]))
    return (
        ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        ),
        plan,
    )
