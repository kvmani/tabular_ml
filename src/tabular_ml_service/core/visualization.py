"""Bounded chart-ready computations for interactive publication-quality plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ..errors import ApiError

MAX_POINTS = 5_000
MAX_CORRELATION_COLUMNS = 50


def _column(frame: pd.DataFrame, name: Any) -> pd.Series:
    if not isinstance(name, str) or name not in frame.columns:
        raise ApiError(
            code="visualization.column_not_found",
            message=f"Column '{name}' does not exist.",
        )
    return frame[name]


def _sample(frame: pd.DataFrame, maximum: int = MAX_POINTS) -> pd.DataFrame:
    return frame.sample(maximum, random_state=42).sort_index() if len(frame) > maximum else frame


def _numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ApiError(
            code="visualization.not_numeric",
            message=f"Column '{name}' has no numeric values.",
        )
    return values


def distribution(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    name = payload.get("column")
    series = _column(frame, name)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(5, int(series.notna().sum() * 0.8)):
        values = numeric.dropna()
        bins = payload.get("bins", "auto")
        if bins == "auto":
            bins = min(60, max(8, int(np.sqrt(len(values)))))
        if not isinstance(bins, int) or not 5 <= bins <= 100:
            raise ApiError(
                code="visualization.invalid_bins",
                message="Histogram bins must be 'auto' or an integer from 5 to 100.",
            )
        counts, edges = np.histogram(values, bins=bins)
        result: dict[str, Any] = {
            "kind": "numeric",
            "column": name,
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "missing": int(series.isna().sum()),
            "statistics": {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            },
        }
        if bool(payload.get("kde", True)) and values.nunique() > 1 and len(values) >= 5:
            grid = np.linspace(float(values.min()), float(values.max()), 200)
            try:
                density = gaussian_kde(values.to_numpy(dtype=float))(grid)
                result["kde"] = {"x": grid.tolist(), "y": density.tolist()}
            except np.linalg.LinAlgError:
                result["kde"] = None
        return result
    counts = series.fillna("(missing)").astype(str).value_counts().head(30)
    return {
        "kind": "categorical",
        "column": name,
        "labels": counts.index.tolist(),
        "counts": [int(value) for value in counts],
        "truncated": series.nunique(dropna=False) > 30,
    }


def correlation(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("columns")
    if selected is None:
        columns = frame.select_dtypes(include="number").columns.tolist()
    elif isinstance(selected, list) and all(isinstance(item, str) for item in selected):
        unknown = sorted(set(selected) - set(frame.columns))
        if unknown:
            raise ApiError(
                code="visualization.column_not_found",
                message="Some correlation columns do not exist.",
                details={"columns": unknown},
            )
        columns = selected
    else:
        raise ApiError(
            code="visualization.invalid_columns",
            message="Correlation columns must be a list.",
        )
    if len(columns) < 2:
        raise ApiError(
            code="visualization.too_few_columns",
            message="Choose at least two numeric columns for correlation.",
        )
    if len(columns) > MAX_CORRELATION_COLUMNS:
        raise ApiError(
            code="visualization.too_many_columns",
            message=f"Correlation plots are limited to {MAX_CORRELATION_COLUMNS} columns.",
        )
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in numeric if numeric[column].notna().sum() >= 3]
    if len(usable) < 2:
        raise ApiError(
            code="visualization.too_few_columns",
            message="At least two selected columns need numeric data.",
        )
    method = payload.get("method", "pearson")
    if method not in {"pearson", "spearman", "kendall"}:
        raise ApiError(
            code="visualization.invalid_method",
            message="Correlation method must be pearson, spearman, or kendall.",
        )
    matrix = numeric[usable].corr(method=method).fillna(0)
    return {
        "labels": usable,
        "matrix": matrix.to_numpy().tolist(),
        "method": method,
    }


def scatter(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    x_name = payload.get("x")
    y_name = payload.get("y")
    x = _numeric(_column(frame, x_name), str(x_name))
    y = _numeric(_column(frame, y_name), str(y_name))
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    color_name = payload.get("color")
    if isinstance(color_name, str):
        color = _column(frame, color_name).reindex(aligned.index).fillna("(missing)").astype(str)
        aligned["color"] = color
    sampled = _sample(aligned)
    trend = None
    if bool(payload.get("trendline", True)) and len(aligned) >= 3:
        slope, intercept = np.polyfit(aligned["x"], aligned["y"], 1)
        predicted = slope * aligned["x"] + intercept
        residual = float(((aligned["y"] - predicted) ** 2).sum())
        total = float(((aligned["y"] - aligned["y"].mean()) ** 2).sum())
        trend = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": 1 - residual / total if total else 0.0,
            "x": [float(aligned["x"].min()), float(aligned["x"].max())],
            "y": [
                float(slope * aligned["x"].min() + intercept),
                float(slope * aligned["x"].max() + intercept),
            ],
        }
    return {
        "x": sampled["x"].tolist(),
        "y": sampled["y"].tolist(),
        "color": sampled["color"].tolist() if "color" in sampled else None,
        "x_label": x_name,
        "y_label": y_name,
        "color_label": color_name if isinstance(color_name, str) else None,
        "sampled": len(sampled) < len(aligned),
        "total_points": len(aligned),
        "trendline": trend,
    }


def box(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    name = payload.get("column")
    values = pd.to_numeric(_column(frame, name), errors="coerce")
    group_name = payload.get("group")
    working = pd.DataFrame({"value": values})
    if isinstance(group_name, str):
        working["group"] = _column(frame, group_name).fillna("(missing)").astype(str)
    else:
        working["group"] = "All rows"
    working = working.dropna(subset=["value"])
    groups = []
    for label, subset in working.groupby("group", sort=True):
        sampled = _sample(subset, 1_000)
        groups.append(
            {
                "label": str(label),
                "values": sampled["value"].tolist(),
                "count": len(subset),
                "sampled": len(sampled) < len(subset),
            }
        )
        if len(groups) >= 20:
            break
    return {
        "column": name,
        "group": group_name if isinstance(group_name, str) else None,
        "groups": groups,
        "truncated_groups": working["group"].nunique() > 20,
    }


def missingness(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.isna().sum().sort_values(ascending=False)
    row_counts = frame.isna().sum(axis=1).value_counts().sort_index()
    return {
        "columns": [str(column) for column in counts.index],
        "counts": [int(value) for value in counts],
        "ratios": [float(value / len(frame)) for value in counts],
        "row_missing_counts": [int(value) for value in row_counts.index],
        "row_frequencies": [int(value) for value in row_counts],
    }


def visualization_payload(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    chart_type = payload.get("type")
    if chart_type == "distribution":
        return {"type": chart_type, "data": distribution(frame, payload)}
    if chart_type == "correlation":
        return {"type": chart_type, "data": correlation(frame, payload)}
    if chart_type == "scatter":
        return {"type": chart_type, "data": scatter(frame, payload)}
    if chart_type == "box":
        return {"type": chart_type, "data": box(frame, payload)}
    if chart_type == "missingness":
        return {"type": chart_type, "data": missingness(frame)}
    raise ApiError(
        code="visualization.invalid_type",
        message="Choose distribution, correlation, scatter, box, or missingness.",
    )
