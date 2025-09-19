"""Modeling and evaluation routes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from config import settings

from backend.app.api import schemas
from backend.app.services import evaluation as eval_utils
from backend.app.services import visualization
from backend.app.services.data_manager import data_manager
from backend.app.services.model_training import TrainingResult, model_trainer
from backend.app.services.run_tracker import RunSummary, run_tracker

router = APIRouter(prefix="/model", tags=["modeling"])


@router.get("/algorithms", response_model=schemas.AlgorithmListResponse)
def algorithms() -> schemas.AlgorithmListResponse:
    algos = [
        schemas.AlgorithmModel(**algo) for algo in model_trainer.available_algorithms()
    ]
    return schemas.AlgorithmListResponse(algorithms=algos)


@router.get("/models", response_model=schemas.ModelListResponse)
def list_models() -> schemas.ModelListResponse:
    model_entries = []
    for model_info in data_manager.list_models():
        model_entries.append(
            schemas.ModelSummary(
                model_id=model_info["model_id"],
                algorithm=model_info["algorithm"],
                task_type=model_info["task_type"],
                target_column=model_info["target_column"],
                metrics=model_info.get("metrics", {}),
                created_at=model_info["created_at"],
            )
        )
    return schemas.ModelListResponse(models=model_entries)


@router.post("/train", response_model=schemas.TrainingResponse)
def train(request: schemas.TrainRequest) -> schemas.TrainingResponse:
    try:
        df = data_manager.get_dataset(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if df.shape[0] > settings.limits.max_rows_train:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {df.shape[0]} rows which exceeds max_rows_train={settings.limits.max_rows_train}",
        )
    split = None
    if request.split_id:
        try:
            split = data_manager.get_split(request.split_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    timeout = settings.ml.timeout_sec

    def _run_training() -> TrainingResult:
        return model_trainer.train(
            dataset_id=request.dataset_id,
            df=df,
            target_column=request.target_column,
            task_type=request.task_type,
            algorithm=request.algorithm,
            hyperparameters=request.hyperparameters,
            split=split,
        )

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_training)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise HTTPException(
            status_code=504,
            detail=f"Training exceeded timeout_sec={timeout}",
        ) from exc
    except ValueError as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=False)

    data_manager.store_split(result.split)
    data_manager.store_model(result.artifact)
    metadata = data_manager.get_metadata(request.dataset_id)
    run_tracker.record(
        RunSummary(
            run_id=result.artifact.model_id,
            dataset_id=request.dataset_id,
            dataset_name=metadata.name,
            algorithm=request.algorithm,
            task_type=request.task_type,
            metrics=result.artifact.metrics,
            created_at=result.artifact.created_at,
        )
    )
    return schemas.TrainingResponse(
        model_id=result.artifact.model_id,
        metrics=result.artifact.metrics,
        history=result.artifact.history,
        split_id=result.split.split_id,
    )


@router.post("/evaluate", response_model=schemas.EvaluationResponse)
def evaluate(request: schemas.EvaluationRequest) -> schemas.EvaluationResponse:
    try:
        model_artifact = data_manager.get_model(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        split = data_manager.get_split(model_artifact.split_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Associated data split not found"
        ) from exc

    X_test = split.X_test.reset_index(drop=True)
    y_test = split.y_test.reset_index(drop=True)
    preprocessor = model_artifact.model_object.get("preprocessor")
    timeout = settings.ml.timeout_sec

    def _evaluate() -> Dict[str, object]:
        if preprocessor is not None:
            X_test_proc = preprocessor.transform(X_test)
        else:
            X_test_proc = X_test.to_numpy()
        preds, proba = model_trainer._predict(model_artifact.model_object, X_test_proc)

        metrics = model_artifact.metrics
        confusion_plot: Optional[Dict[str, object]] = None
        roc_plot: Optional[Dict[str, object]] = None
        regression_plots: Optional[Dict[str, object]] = None

        if model_artifact.task_type == "classification":
            confusion = eval_utils.confusion_matrix_data(y_test, preds)
            confusion_plot = visualization.confusion_matrix_plot(confusion)
            if proba is not None:
                roc_data = eval_utils.roc_curve_data(y_test, proba)
                roc_plot = visualization.roc_curve_plot(roc_data)
        else:
            diagnostics = eval_utils.regression_diagnostics(y_test, preds)
            regression_plots = {
                "comparison": visualization.regression_comparison_plot(diagnostics),
                "residuals": visualization.residuals_plot(diagnostics),
            }

        history_plot = visualization.training_history(model_artifact.history)
        return {
            "metrics": metrics,
            "confusion_matrix": confusion_plot,
            "roc_curve": roc_plot,
            "regression_diagnostics": regression_plots,
            "training_history": history_plot,
        }

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_evaluate)
    try:
        payload = future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise HTTPException(
            status_code=504,
            detail=f"Evaluation exceeded timeout_sec={timeout}",
        ) from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=False)

    return schemas.EvaluationResponse(**payload)


