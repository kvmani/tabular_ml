"""Semantic schema inference, quality analysis, and target recommendations."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

SemanticType = Literal[
    "empty",
    "constant",
    "boolean",
    "integer",
    "continuous",
    "datetime",
    "categorical",
    "text",
    "identifier",
]
FeatureRole = Literal["feature", "identifier", "ignore"]

IDENTIFIER_NAME = re.compile(r"(^id$|_id$|^uuid$|identifier|record.?number)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    storage_type: str
    semantic_type: SemanticType
    suggested_role: FeatureRole
    non_missing: int
    missing: int
    missing_ratio: float
    unique: int
    unique_ratio: float
    sample_values: tuple[Any, ...]
    statistics: dict[str, Any]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sample_values"] = list(self.sample_values)
        result["warnings"] = list(self.warnings)
        return result


def _sample_values(series: pd.Series, maximum: int = 5) -> tuple[Any, ...]:
    values: list[Any] = []
    for value in series.dropna().unique()[:maximum]:
        if isinstance(value, (np.integer, np.floating)):
            values.append(value.item())
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            values.append(pd.Timestamp(value).isoformat())
        else:
            values.append(str(value) if not isinstance(value, (str, bool, int, float)) else value)
    return tuple(values)


def _looks_datetime(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if len(non_missing) < 3:
        return False
    sample = non_missing.astype(str).head(200)
    marker = (
        r"\d{1,4}[-/:]\d{1,2}|"
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    )
    has_date_markers = sample.str.contains(marker, case=False, regex=True).mean() >= 0.8
    if not has_date_markers:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", utc=True, format="mixed")
    return float(parsed.notna().mean()) >= 0.9


def _semantic_type(name: str, series: pd.Series) -> SemanticType:
    non_missing = series.dropna()
    unique = int(non_missing.nunique())
    if non_missing.empty:
        return "empty"
    if unique == 1:
        return "constant"
    normalized = {str(value).strip().lower() for value in non_missing.unique()[:20]}
    if pd.api.types.is_bool_dtype(series) or normalized <= {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "0",
        "1",
    }:
        return "boolean"
    unique_ratio = unique / len(non_missing)
    if IDENTIFIER_NAME.search(name) and unique_ratio > 0.8:
        return "identifier"
    if pd.api.types.is_datetime64_any_dtype(series) or (
        pd.api.types.is_object_dtype(series) and _looks_datetime(series)
    ):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        return "continuous"
    mean_length = float(non_missing.astype(str).str.len().mean())
    if mean_length > 32 or (unique_ratio > 0.7 and unique > 40):
        return "text"
    if unique_ratio > 0.98 and len(non_missing) >= 30:
        return "identifier"
    return "categorical"


def _statistics(series: pd.Series, semantic_type: SemanticType) -> dict[str, Any]:
    non_missing = series.dropna()
    if semantic_type in {"integer", "continuous"}:
        numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
        if numeric.empty:
            return {}
        return {
            "mean": float(numeric.mean()),
            "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
            "min": float(numeric.min()),
            "q1": float(numeric.quantile(0.25)),
            "median": float(numeric.median()),
            "q3": float(numeric.quantile(0.75)),
            "max": float(numeric.max()),
            "skew": float(cast(float, numeric.skew())) if len(numeric) > 2 else 0.0,
        }
    counts = non_missing.astype(str).value_counts().head(8)
    return {
        "top_values": [
            {"value": str(value), "count": int(count), "ratio": float(count / len(non_missing))}
            for value, count in counts.items()
        ]
    }


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    non_missing = int(series.notna().sum())
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    semantic = _semantic_type(name, series)
    warnings: list[str] = []
    missing_ratio = missing / len(series)
    unique_ratio = unique / non_missing if non_missing else 0.0
    if missing_ratio >= 0.4:
        warnings.append("high_missingness")
    elif missing:
        warnings.append("has_missing_values")
    if semantic == "constant":
        warnings.append("constant_column")
    if semantic == "identifier":
        warnings.append("identifier_like")
    if semantic == "categorical" and unique > 50:
        warnings.append("high_cardinality")
    role: FeatureRole = (
        "identifier"
        if semantic == "identifier"
        else "ignore"
        if semantic in {"empty", "constant", "text"}
        else "feature"
    )
    return ColumnProfile(
        name=name,
        storage_type=str(series.dtype),
        semantic_type=semantic,
        suggested_role=role,
        non_missing=non_missing,
        missing=missing,
        missing_ratio=missing_ratio,
        unique=unique,
        unique_ratio=unique_ratio,
        sample_values=_sample_values(series),
        statistics=_statistics(series, semantic),
        warnings=tuple(warnings),
    )


def _target_candidate(profile: ColumnProfile, row_count: int) -> dict[str, Any] | None:
    if profile.semantic_type in {"empty", "constant", "identifier", "text", "datetime"}:
        return None
    if profile.missing_ratio >= 0.5 or profile.non_missing < 20:
        return None
    classification_limit = max(12, min(50, int(math.sqrt(row_count)) + 2))
    classification = profile.semantic_type in {"boolean", "categorical"} or (
        profile.semantic_type == "integer" and profile.unique <= classification_limit
    )
    task = "classification" if classification else "regression"
    score = 60
    if profile.name.lower() in {"target", "label", "class", "outcome", "response", "survived"}:
        score += 25
    if task == "classification" and 2 <= profile.unique <= 12:
        score += 10
    if profile.semantic_type == "boolean":
        score += 8
    if profile.missing == 0:
        score += 5
    return {
        "column": profile.name,
        "task": task,
        "confidence": min(score, 100),
        "reason": (
            f"{profile.unique} distinct outcome levels"
            if task == "classification"
            else "numeric values with enough variation for regression"
        ),
    }


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a bounded, frontend-ready dataset profile."""
    profiles = [profile_column(str(name), frame[name]) for name in frame.columns]
    missing_cells = int(frame.isna().sum().sum())
    total_cells = max(int(frame.size), 1)
    duplicate_rows = int(frame.duplicated().sum())
    constant_count = sum(item.semantic_type in {"constant", "empty"} for item in profiles)
    identifier_count = sum(item.semantic_type == "identifier" for item in profiles)
    warnings: list[dict[str, Any]] = []
    if missing_cells:
        warnings.append(
            {
                "code": "missing_values",
                "severity": "warning",
                "message": f"{missing_cells:,} cells are missing and need an explicit strategy.",
            }
        )
    if duplicate_rows:
        warnings.append(
            {
                "code": "duplicate_rows",
                "severity": "warning",
                "message": f"{duplicate_rows:,} duplicate rows may bias validation.",
            }
        )
    if constant_count:
        warnings.append(
            {
                "code": "constant_columns",
                "severity": "info",
                "message": (
                    f"{constant_count} constant or empty columns will be excluded by default."
                ),
            }
        )
    if identifier_count:
        warnings.append(
            {
                "code": "identifier_columns",
                "severity": "info",
                "message": (
                    f"{identifier_count} identifier-like columns are excluded to reduce leakage."
                ),
            }
        )
    targets = [
        candidate
        for item in profiles
        if (candidate := _target_candidate(item, len(frame))) is not None
    ]
    targets.sort(key=lambda item: (-int(item["confidence"]), str(item["column"])))
    missing_penalty = min(35.0, missing_cells / total_cells * 100)
    duplicate_penalty = min(20.0, duplicate_rows / len(frame) * 100 if len(frame) else 0)
    structure_penalty = min(20.0, (constant_count + identifier_count) / len(profiles) * 20)
    health = max(0, round(100 - missing_penalty - duplicate_penalty - structure_penalty))
    return {
        "overview": {
            "rows": len(frame),
            "columns": len(frame.columns),
            "missing_cells": missing_cells,
            "missing_ratio": missing_cells / total_cells,
            "duplicate_rows": duplicate_rows,
            "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
            "health_score": health,
        },
        "columns": [item.as_dict() for item in profiles],
        "target_candidates": targets,
        "warnings": warnings,
    }
