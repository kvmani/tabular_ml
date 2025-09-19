"""Preprocessing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api import schemas
from backend.app.services import preprocess
from backend.app.services.data_manager import data_manager
from backend.app.api.utils import metadata_to_model

router = APIRouter(prefix="/preprocess", tags=["preprocess"])


def _get_dataframe(dataset_id: str):
    try:
        return data_manager.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{dataset_id}/detect-outliers", response_model=schemas.OutlierDetectionResponse
)
def detect_outliers(
    dataset_id: str, request: schemas.OutlierDetectionRequest
) -> schemas.OutlierDetectionResponse:
    df = _get_dataframe(dataset_id)
    report = preprocess.detect_outliers(
        df,
        columns=request.columns,
        z_threshold=request.z_threshold,
    )
    return schemas.OutlierDetectionResponse(**report.to_dict())


@router.post(
    "/{dataset_id}/remove-outliers", response_model=schemas.OutlierRemovalResponse
)
def remove_outliers(
    dataset_id: str, request: schemas.OutlierDetectionRequest
) -> schemas.OutlierRemovalResponse:
    df = _get_dataframe(dataset_id)
    report = preprocess.detect_outliers(
        df,
        columns=request.columns,
        z_threshold=request.z_threshold,
    )
    cleaned = preprocess.remove_outliers(
        df,
        columns=request.columns,
        z_threshold=request.z_threshold,
    )
    try:
        metadata = data_manager.create_dataset(
            cleaned,
            name=f"{data_manager.get_metadata(dataset_id).name} (outliers removed)",
            source=f"derived:{dataset_id}",
            description="Dataset with outliers removed",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.OutlierRemovalResponse(
        dataset=metadata_to_model(metadata),
        report=schemas.OutlierDetectionResponse(**report.to_dict()),
    )


@router.post("/{dataset_id}/impute", response_model=schemas.ImputationResponse)
def impute(
    dataset_id: str, request: schemas.ImputationRequest
) -> schemas.ImputationResponse:
    df = _get_dataframe(dataset_id)
    imputed = preprocess.impute_missing(
        df,
        strategy=request.strategy,
        columns=request.columns,
        fill_value=request.fill_value,
    )
    try:
        metadata = data_manager.create_dataset(
            imputed,
            name=f"{data_manager.get_metadata(dataset_id).name} (imputed)",
            source=f"derived:{dataset_id}",
            description="Dataset with missing values imputed",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.ImputationResponse(dataset=metadata_to_model(metadata))


@router.post("/{dataset_id}/filter", response_model=schemas.FilterResponse)
def filter_dataset(
    dataset_id: str, request: schemas.FilterRequest
) -> schemas.FilterResponse:
    df = _get_dataframe(dataset_id)
    try:
        filtered = preprocess.apply_filters(df, [rule.dict() for rule in request.rules])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        metadata = data_manager.create_dataset(
            filtered,
            name=f"{data_manager.get_metadata(dataset_id).name} (filtered)",
            source=f"derived:{dataset_id}",
            description="Filtered dataset",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.FilterResponse(dataset=metadata_to_model(metadata))


@router.post("/{dataset_id}/split", response_model=schemas.SplitResponse)
def split(dataset_id: str, request: schemas.SplitRequest) -> schemas.SplitResponse:
    df = _get_dataframe(dataset_id)
    split_obj = preprocess.split_dataset(
        df,
        target_column=request.target_column,
        task_type=request.task_type,
        test_size=request.test_size,
        val_size=request.val_size,
        random_state=request.random_state,
        stratify=request.stratify,
    )
    split_obj.dataset_id = dataset_id
    data_manager.store_split(split_obj)
    return schemas.SplitResponse(
        split_id=split_obj.split_id,
        dataset_id=dataset_id,
        target_column=split_obj.target_column,
        feature_columns=split_obj.feature_columns,
    )
