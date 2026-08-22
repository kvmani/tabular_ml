"""Leakage-safe training, evaluation, diagnostics, importance, and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from ..errors import ApiError
from .algorithms import AlgorithmSpec, Task, estimator_for
from .preprocessing import PreparationConfig, build_preprocessor


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    test_size: float = 0.2
    cv_folds: int = 5
    random_seed: int = 42
    permutation_importance: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> TrainingOptions:
        raw = payload or {}
        test_size = float(raw.get("test_size", 0.2))
        cv_folds = int(raw.get("cv_folds", 5))
        random_seed = int(raw.get("random_seed", 42))
        if not 0.1 <= test_size <= 0.4:
            raise ApiError(
                code="training.invalid_options",
                message="Test fraction must be between 0.1 and 0.4.",
            )
        if not 2 <= cv_folds <= 10:
            raise ApiError(
                code="training.invalid_options",
                message="Cross-validation folds must be between 2 and 10.",
            )
        if random_seed < 0:
            raise ApiError(
                code="training.invalid_options",
                message="Random seed cannot be negative.",
            )
        return cls(
            test_size=test_size,
            cv_folds=cv_folds,
            random_seed=random_seed,
            permutation_importance=bool(raw.get("permutation_importance", True)),
        )


class DenseGuardTransformer(TransformerMixin, BaseEstimator):
    """Convert sparse prepared features to dense with an explicit memory guard."""

    def __init__(self, max_cells: int = 25_000_000) -> None:
        self.max_cells = max_cells

    def fit(self, values: Any, y: Any = None) -> DenseGuardTransformer:
        return self

    def transform(self, values: Any) -> np.ndarray:
        cells = int(values.shape[0]) * int(values.shape[1])
        if cells > self.max_cells:
            raise ApiError(
                code="training.dense_limit",
                message=(
                    "This algorithm requires a dense feature matrix that would be too large. "
                    "Use ordinal encoding, exclude high-cardinality columns, "
                    "or choose another model."
                ),
            )
        if sparse.issparse(values):
            return np.asarray(cast(Any, values).toarray())
        return np.asarray(values)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features)


@dataclass(slots=True)
class FittedRun:
    pipeline: Pipeline
    target: str
    task: Task
    algorithm: AlgorithmSpec
    parameters: dict[str, Any]
    preparation: PreparationConfig
    options: TrainingOptions
    feature_columns: list[str]
    result: dict[str, Any]


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _finite_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _validate_target(
    y: pd.Series,
    task: Task,
    test_size: float,
) -> list[Any] | None:
    if len(y) < 20:
        raise ApiError(
            code="training.too_few_rows",
            message="At least 20 rows with a known target are required for training.",
        )
    if task == "regression":
        numeric = pd.to_numeric(y, errors="coerce")
        if numeric.isna().any():
            raise ApiError(
                code="target.non_numeric",
                message="Regression requires a numeric target with no unparseable values.",
            )
        if numeric.nunique() < 3:
            raise ApiError(
                code="target.low_variation",
                message="Regression requires at least three distinct target values.",
            )
        return None
    counts = y.value_counts()
    if len(counts) < 2:
        raise ApiError(
            code="target.one_class",
            message="Classification requires at least two target classes.",
        )
    if len(counts) > 50:
        raise ApiError(
            code="target.too_many_classes",
            message="Classification is limited to 50 target classes.",
        )
    if int(counts.min()) < 2:
        raise ApiError(
            code="target.rare_class",
            message="Every target class needs at least two rows.",
        )
    test_rows = math.ceil(len(y) * test_size)
    if test_rows < len(counts):
        raise ApiError(
            code="training.test_too_small",
            message=(
                "The test split is too small to contain every class. "
                "Increase the test fraction or combine rare classes."
            ),
        )
    return counts.index.tolist()


def _cv_configuration(
    y_train: pd.Series,
    task: Task,
    requested_folds: int,
    random_seed: int,
) -> tuple[Any, int, dict[str, str]]:
    if task == "classification":
        effective = min(requested_folds, int(y_train.value_counts().min()))
        if effective < 2:
            raise ApiError(
                code="training.cv_impossible",
                message="The training split has too few examples in its rarest class.",
            )
        splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=random_seed)
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
        }
    else:
        effective = min(requested_folds, len(y_train))
        splitter = KFold(n_splits=effective, shuffle=True, random_state=random_seed)
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }
    return splitter, effective, scoring


def _cross_validation_summary(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: Task,
    options: TrainingOptions,
) -> dict[str, Any]:
    splitter, folds, scoring = _cv_configuration(
        y_train,
        task,
        options.cv_folds,
        options.random_seed,
    )
    try:
        raw = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            code="training.cross_validation_failed",
            message=(
                "Cross-validation failed for this model and preparation combination. "
                "Review missing values, feature types, and algorithm settings."
            ),
            details={"exception": type(exc).__name__},
        ) from exc
    metrics: dict[str, Any] = {}
    for name in scoring:
        values = np.asarray(raw[f"test_{name}"], dtype=float)
        if name in {"rmse", "mae"}:
            values = -values
        metrics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "fold_scores": values.tolist(),
        }
    return {
        "folds": folds,
        "metrics": metrics,
        "fit_time_mean": float(np.asarray(raw["fit_time"]).mean()),
        "score_time_mean": float(np.asarray(raw["score_time"]).mean()),
    }


def _classification_evaluation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = pipeline.named_steps["model"]
    classes = list(model.classes_)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "matthews_correlation": float(matthews_corrcoef(y_test, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_test, predictions)),
    }
    labels = [_safe_scalar(item) for item in classes]
    diagnostics: dict[str, Any] = {
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }
    probabilities: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(X_test))
        metrics["log_loss"] = float(log_loss(y_test, probabilities, labels=classes))
        if len(classes) > 2:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    if len(classes) == 2:
        positive = classes[-1]
        if probabilities is not None:
            score = probabilities[:, -1]
            metrics["brier_score"] = float(
                brier_score_loss((y_test == positive).astype(int), score)
            )
        elif hasattr(pipeline, "decision_function"):
            score = np.asarray(pipeline.decision_function(X_test))
        else:
            score = None
        if score is not None:
            fpr, tpr, roc_thresholds = roc_curve(y_test, score, pos_label=positive)
            precision, recall, pr_thresholds = precision_recall_curve(
                y_test,
                score,
                pos_label=positive,
            )
            metrics["roc_auc"] = float(roc_auc_score(y_test == positive, score))
            diagnostics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": _finite_list(roc_thresholds),
                "positive_class": _safe_scalar(positive),
            }
            diagnostics["precision_recall_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": _finite_list(pr_thresholds),
                "positive_class": _safe_scalar(positive),
            }
    prediction_rows = []
    for position, (index, actual, predicted) in enumerate(
        zip(X_test.index, y_test, predictions, strict=True)
    ):
        row = {
            "row": str(index),
            "actual": _safe_scalar(actual),
            "predicted": _safe_scalar(predicted),
        }
        if probabilities is not None:
            row["probabilities"] = {
                str(_safe_scalar(label)): float(value)
                for label, value in zip(classes, probabilities[position], strict=True)
            }
        prediction_rows.append(row)
        if len(prediction_rows) >= 500:
            break
    diagnostics["predictions"] = prediction_rows
    return metrics, diagnostics


def _regression_evaluation(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = pd.to_numeric(y_test, errors="raise").to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    residuals = actual - predicted
    nonzero = actual != 0
    metrics = {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "explained_variance": float(explained_variance_score(actual, predicted)),
        "mape": (
            float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])))
            if nonzero.any()
            else None
        ),
    }
    points = [
        {
            "row": str(index),
            "actual": float(observed),
            "predicted": float(estimate),
            "residual": float(residual),
        }
        for index, observed, estimate, residual in zip(
            X_test.index[:500],
            actual[:500],
            predicted[:500],
            residuals[:500],
            strict=True,
        )
    ]
    diagnostics = {
        "predictions": points,
        "residual_summary": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0,
            "q05": float(np.quantile(residuals, 0.05)),
            "q25": float(np.quantile(residuals, 0.25)),
            "median": float(np.median(residuals)),
            "q75": float(np.quantile(residuals, 0.75)),
            "q95": float(np.quantile(residuals, 0.95)),
        },
    }
    return metrics, diagnostics


def _native_importance(pipeline: Pipeline) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    values: np.ndarray | None = None
    kind = ""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = (
            np.abs(coefficients)
            if coefficients.ndim == 1
            else np.mean(np.abs(coefficients), axis=0)
        )
        kind = "coefficient_magnitude"
    if values is None:
        return []
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        names = [f"feature_{index}" for index in range(len(values))]
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    ranked = sorted(zip(names, values, strict=True), key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": str(name), "importance": float(value), "kind": kind}
        for name, value in ranked[:50]
    ]


def _permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: Task,
    random_seed: int,
) -> list[dict[str, Any]]:
    if len(X_test) > 1_000:
        sample = X_test.sample(1_000, random_state=random_seed)
        target = y_test.loc[sample.index]
    else:
        sample = X_test
        target = y_test
    scoring = "balanced_accuracy" if task == "classification" else "r2"
    try:
        result = permutation_importance(
            pipeline,
            sample,
            target,
            scoring=scoring,
            n_repeats=3,
            random_state=random_seed,
            n_jobs=1,
        )
    except Exception:
        return []
    ranked = sorted(
        zip(sample.columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": str(name),
            "importance": float(mean),
            "std": float(std),
            "kind": "permutation",
        }
        for name, mean, std in ranked[:50]
    ]


def train_model(
    frame: pd.DataFrame,
    *,
    target: str,
    algorithm_id: str,
    parameters: dict[str, Any] | None,
    preparation: PreparationConfig,
    options: TrainingOptions,
) -> FittedRun:
    transformer, plan = build_preprocessor(frame, target, preparation)
    task: Task = plan["task"]
    y = frame[target]
    known_target = y.notna()
    X = frame.loc[known_target].drop(columns=[target])
    y = y.loc[known_target]
    _validate_target(y, task, options.test_size)
    if task == "regression":
        y = pd.to_numeric(y, errors="raise")
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=options.test_size,
        random_state=options.random_seed,
        stratify=stratify,
    )
    spec, estimator, validated = estimator_for(
        algorithm_id,
        task,
        parameters,
        random_seed=options.random_seed,
    )
    steps: list[tuple[str, Any]] = [("preprocess", transformer)]
    if spec.requires_dense:
        steps.append(("dense", DenseGuardTransformer()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    cross_validation = _cross_validation_summary(pipeline, X_train, y_train, task, options)
    try:
        pipeline.fit(X_train, y_train)
        predictions = np.asarray(pipeline.predict(X_test))
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            code="training.fit_failed",
            message=(
                "The model could not be fitted with this data and configuration. "
                "Try the recommended defaults or inspect the preparation plan."
            ),
            details={"exception": type(exc).__name__},
        ) from exc
    if task == "classification":
        metrics, diagnostics = _classification_evaluation(
            pipeline,
            X_test,
            y_test,
            predictions,
        )
        primary_metric = "balanced_accuracy"
    else:
        metrics, diagnostics = _regression_evaluation(X_test, y_test, predictions)
        primary_metric = "r2"
    importance = {
        "native": _native_importance(pipeline),
        "permutation": (
            _permutation_importance(
                pipeline,
                X_test,
                y_test,
                task,
                options.random_seed,
            )
            if options.permutation_importance
            else []
        ),
    }
    result = {
        "task": task,
        "target": target,
        "algorithm": spec.public_dict(),
        "parameters": validated,
        "preparation": asdict(preparation),
        "options": asdict(options),
        "split": {
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "excluded_missing_target_rows": int((~known_target).sum()),
            "random_seed": options.random_seed,
        },
        "primary_metric": primary_metric,
        "metrics": metrics,
        "cross_validation": cross_validation,
        "diagnostics": diagnostics,
        "importance": importance,
    }
    return FittedRun(
        pipeline=pipeline,
        target=target,
        task=task,
        algorithm=spec,
        parameters=validated,
        preparation=preparation,
        options=options,
        feature_columns=X.columns.tolist(),
        result=result,
    )


def predict_records(run: FittedRun, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ApiError(code="inference.no_rows", message="Provide at least one row for prediction.")
    if len(records) > 1_000:
        raise ApiError(
            code="inference.too_many_rows",
            message="Inference is limited to 1,000 rows per request.",
            status=413,
        )
    frame = pd.DataFrame(records)
    missing = sorted(set(run.feature_columns) - set(frame.columns))
    if missing:
        raise ApiError(
            code="inference.missing_columns",
            message="Prediction rows are missing required columns.",
            details={"columns": missing},
        )
    frame = frame.loc[:, run.feature_columns]
    try:
        predictions = np.asarray(run.pipeline.predict(frame))
        probabilities = (
            np.asarray(run.pipeline.predict_proba(frame))
            if run.task == "classification" and hasattr(run.pipeline, "predict_proba")
            else None
        )
    except Exception as exc:
        raise ApiError(
            code="inference.failed",
            message="Prediction failed. Check that values match the training feature types.",
            details={"exception": type(exc).__name__},
        ) from exc
    classes = list(run.pipeline.named_steps["model"].classes_) if probabilities is not None else []
    output = []
    for index, prediction in enumerate(predictions):
        row: dict[str, Any] = {"prediction": _safe_scalar(prediction)}
        if probabilities is not None:
            row["probabilities"] = {
                str(_safe_scalar(label)): float(value)
                for label, value in zip(classes, probabilities[index], strict=True)
            }
        output.append(row)
    return {"rows": output, "task": run.task, "target": run.target}
