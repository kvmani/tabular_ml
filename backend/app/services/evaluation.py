"""Evaluation helpers for trained models."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn import metrics


logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return standard classification metrics."""

    result = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(
            metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    unique_classes = np.unique(list(y_true))
    if proba is not None and len(unique_classes) == 2:
        # Assume positive class is the last label
        if proba.ndim == 2 and proba.shape[1] >= 2:
            result["roc_auc"] = float(metrics.roc_auc_score(y_true, proba[:, -1]))
        else:
            result["roc_auc"] = float(metrics.roc_auc_score(y_true, proba))
    logger.info(
        "Computed classification metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        result.get("accuracy", 0.0),
        result.get("precision", 0.0),
        result.get("recall", 0.0),
        result.get("f1", 0.0),
    )
    return result


def regression_metrics(y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
    """Return regression metrics for model assessment."""

    result = {
        "r2": float(metrics.r2_score(y_true, y_pred)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "mse": float(metrics.mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
    }
    logger.info(
        "Computed regression metrics: r2=%.4f mae=%.4f rmse=%.4f",
        result["r2"],
        result["mae"],
        result["rmse"],
    )
    return result


def confusion_matrix_data(y_true: Iterable, y_pred: Iterable) -> Dict[str, object]:
    """Produce confusion matrix data for visualization."""

    labels = np.unique(list(y_true) + list(y_pred))
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    logger.info(
        "Generated confusion matrix for %s labels (size=%s).",
        len(labels),
        matrix.shape,
    )
    return {
        "labels": labels.tolist(),
        "matrix": matrix.tolist(),
    }


def roc_curve_data(
    y_true: Iterable,
    proba: np.ndarray,
    positive_label: Optional[str] = None,
) -> Optional[Dict[str, List[float]]]:
    """Generate ROC curve coordinates if supported."""

    unique_classes = np.unique(list(y_true))
    if len(unique_classes) != 2:
        return None
    pos_label = positive_label or unique_classes[-1]
    if proba.ndim == 2:
        y_score = proba[:, -1]
    else:
        y_score = proba
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    logger.info("Generated ROC curve with %s thresholds.", len(thresholds))
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def regression_diagnostics(
    y_true: Iterable, y_pred: Iterable
) -> Dict[str, List[float]]:
    """Return data for regression diagnostic plots."""

    residuals = np.array(y_true) - np.array(y_pred)
    logger.info("Prepared regression diagnostics for %s predictions.", len(residuals))
    return {
        "predicted": list(map(float, y_pred)),
        "residuals": list(map(float, residuals)),
        "actual": list(map(float, y_true)),
    }


def classification_report_text(y_true: Iterable, y_pred: Iterable) -> str:
    """Return a formatted classification report."""

    report = metrics.classification_report(y_true, y_pred, zero_division=0)
    logger.info("Generated classification report (length=%s characters).", len(report))
    return report
