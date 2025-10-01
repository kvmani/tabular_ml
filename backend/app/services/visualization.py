"""Visualization helpers leveraging Plotly for offline rendering."""
from __future__ import annotations

from typing import Dict, List, Optional

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def _to_plotly_json(fig: go.Figure) -> Dict[str, object]:
    """Return a JSON-serialisable representation of a Plotly figure."""
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


COLOR_PALETTE = px.colors.qualitative.D3


def histogram(df: pd.DataFrame, column: str) -> Dict[str, object]:
    fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
    fig.update_layout(template="plotly_white", bargap=0.1)
    return _to_plotly_json(fig)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
) -> Dict[str, object]:
    fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {x} vs {y}")
    fig.update_layout(template="plotly_white")
    return _to_plotly_json(fig)


def training_history(
    history: List[Dict[str, object]], metric: str = "train_loss"
) -> Dict[str, object]:
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="No training history available", template="plotly_white"
        )
        return _to_plotly_json(fig)
    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    fig = go.Figure()
    if any("train_loss" in entry for entry in history):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[entry.get("train_loss") for entry in history],
                mode="lines+markers",
                name="Train Loss",
            )
        )
    if any("val_loss" in entry for entry in history):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[entry.get("val_loss") for entry in history],
                mode="lines+markers",
                name="Validation Loss",
            )
        )
    if any(
        "metrics" in entry and isinstance(entry["metrics"], dict) for entry in history
    ):
        metric_key = metric
        sample_metrics = history[-1]["metrics"]
        if metric_key not in sample_metrics:
            metric_key = next(iter(sample_metrics.keys()))
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[entry.get("metrics", {}).get(metric_key) for entry in history],
                mode="lines+markers",
                name=f"Validation {metric_key}",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=f"Validation {metric_key}",
                overlaying="y",
                side="right",
            )
        )
    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )
    return _to_plotly_json(fig)


def confusion_matrix_plot(data: Dict[str, object]) -> Dict[str, object]:
    labels = data.get("labels", [])
    matrix = data.get("matrix", [])
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
    )
    return _to_plotly_json(fig)


def roc_curve_plot(
    data: Optional[Dict[str, List[float]]]
) -> Optional[Dict[str, object]]:
    if not data:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data["fpr"], y=data["tpr"], mode="lines", name="ROC Curve")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")
        )
    )
    fig.update_layout(
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    return _to_plotly_json(fig)


def regression_comparison_plot(data: Dict[str, List[float]]) -> Dict[str, object]:
    actual = data.get("actual", [])
    predicted = data.get("predicted", [])
    if not actual or not predicted:
        fig = go.Figure()
        fig.update_layout(
            title="Actual vs Predicted (no data)", template="plotly_white"
        )
        return _to_plotly_json(fig)
    fig = px.scatter(
        x=actual,
        y=predicted,
        labels={"x": "Actual", "y": "Predicted"},
        title="Actual vs Predicted",
    )
    min_val = min(actual + predicted)
    max_val = max(actual + predicted)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal",
        )
    )
    fig.update_layout(template="plotly_white")
    return _to_plotly_json(fig)


def residuals_plot(data: Dict[str, List[float]]) -> Dict[str, object]:
    predicted = data.get("predicted", [])
    residuals = data.get("residuals", [])
    if not predicted or not residuals:
        fig = go.Figure()
        fig.update_layout(title="Residual Plot (no data)", template="plotly_white")
        return _to_plotly_json(fig)
    fig = px.scatter(
        x=predicted,
        y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title="Residual Plot",
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white")
    return _to_plotly_json(fig)
