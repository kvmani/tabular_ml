"""Preprocessing utilities for datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from backend.app.models.storage import DataSplit


@dataclass
class OutlierReport:
    """Structured response describing detected outliers."""

    total_outliers: int
    inspected_columns: List[str]
    sample_indices: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_outliers": self.total_outliers,
            "inspected_columns": self.inspected_columns,
            "sample_indices": self.sample_indices,
        }


SUPPORTED_OPERATORS = {
    "eq": lambda series, value: series == value,
    "ne": lambda series, value: series != value,
    "gt": lambda series, value: series > value,
    "gte": lambda series, value: series >= value,
    "lt": lambda series, value: series < value,
    "lte": lambda series, value: series <= value,
    "contains": lambda series, value: series.astype(str).str.contains(
        str(value), na=False
    ),
    "in": lambda series, value: series.isin(
        value if isinstance(value, Iterable) else [value]
    ),
}


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    z_threshold: float = 3.0,
) -> OutlierReport:
    """Identify outliers using a z-score heuristic."""

    numeric_columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return OutlierReport(0, [], [])

    z_scores = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        series = df[column].astype(float)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_scores[column] = 0.0
            continue
        z_scores[column] = (series - series.mean()) / std

    mask = (z_scores.abs() > z_threshold).any(axis=1)
    indices = df.index[mask].tolist()
    return OutlierReport(
        total_outliers=len(indices),
        inspected_columns=numeric_columns,
        sample_indices=indices[:20],
    )


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Drop rows flagged as outliers by :func:`detect_outliers`."""

    report = detect_outliers(df, columns=columns, z_threshold=z_threshold)
    if report.total_outliers == 0:
        return df.copy()
    # Remove all rows that exceed the threshold (recompute mask accurately)
    numeric_columns = report.inspected_columns
    z_scores = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        series = df[column].astype(float)
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            z_scores[column] = 0.0
            continue
        z_scores[column] = (series - series.mean()) / std
    full_mask = (z_scores.abs() > z_threshold).any(axis=1)
    return df.loc[~full_mask].reset_index(drop=True)


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """Impute missing values using scikit-learn strategies."""

    if columns is None:
        columns = df.columns.tolist()
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    subset = df[columns]
    imputed = imputer.fit_transform(subset)
    new_df = df.copy()
    new_df[columns] = imputed
    return new_df


def apply_filters(df: pd.DataFrame, rules: List[Dict[str, object]]) -> pd.DataFrame:
    """Filter rows based on comparison rules."""

    if not rules:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        column = rule.get("column")
        operator = rule.get("operator", "eq")
        value = rule.get("value")
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported operator '{operator}'")
        series = df[column]
        comparator = SUPPORTED_OPERATORS[operator]
        if operator != "contains" and pd.api.types.is_numeric_dtype(series):
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
        mask &= comparator(series, value)
    return df.loc[mask].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> DataSplit:
    """Split a dataframe into train/validation/test segments."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present")
    features = df.drop(columns=[target_column])
    target = df[target_column]

    stratify_data = target if stratify and task_type == "classification" else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_data,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_val = y_train_val if stratify_data is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_val,
    )

    split = DataSplit(
        split_id=uuid4().hex,
        dataset_id="",
        feature_columns=list(features.columns),
        target_column=target_column,
        task_type=task_type,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    return split
