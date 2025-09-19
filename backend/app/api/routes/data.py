"""Dataset management routes."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from config import settings

from backend.app.api import schemas
from backend.app.services.data_manager import data_manager
from backend.app.services.file_loader import read_tabular_file
from backend.app.api.utils import metadata_to_model

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/datasets", response_model=schemas.DatasetListResponse)
def list_datasets() -> schemas.DatasetListResponse:
    datasets = data_manager.list_datasets()
    return schemas.DatasetListResponse(datasets=datasets)


@router.get("/samples", response_model=schemas.SampleDatasetListResponse)
def list_samples() -> schemas.SampleDatasetListResponse:
    samples = data_manager.list_sample_datasets()
    sample_models = [schemas.SampleDatasetModel(**sample) for sample in samples]
    return schemas.SampleDatasetListResponse(samples=sample_models)


@router.post("/samples/{sample_key}", response_model=schemas.UploadResponse)
def load_sample(sample_key: str) -> schemas.UploadResponse:
    try:
        metadata = data_manager.load_sample_dataset(sample_key)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.UploadResponse(dataset=metadata_to_model(metadata))


if settings.app.allow_file_uploads:
    from fastapi import File, Form, UploadFile  # type: ignore

    @router.post("/upload", response_model=schemas.UploadResponse)
    async def upload_dataset(  # pragma: no cover - optional feature
        file: UploadFile = File(...),
        name: str = Form("Uploaded Dataset"),
        description: str = Form("User uploaded dataset"),
    ) -> schemas.UploadResponse:
        contents = await file.read()
        try:
            df = read_tabular_file(contents, file.filename)
            metadata = data_manager.create_dataset(
                df=df,
                name=name or file.filename,
                source=f"upload:{file.filename}",
                description=description,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return schemas.UploadResponse(dataset=metadata_to_model(metadata))


@router.get("/{dataset_id}/preview", response_model=schemas.PreviewResponse)
def dataset_preview(
    dataset_id: str, limit: int = Query(50, ge=1)
) -> schemas.PreviewResponse:
    if limit > settings.limits.max_rows_preview:
        raise HTTPException(
            status_code=400,
            detail=f"Preview limit {limit} exceeds max_rows_preview={settings.limits.max_rows_preview}",
        )
    try:
        df = data_manager.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    preview = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return schemas.PreviewResponse(data=preview)


@router.get("/{dataset_id}/summary", response_model=schemas.SummaryResponse)
def dataset_summary(dataset_id: str) -> schemas.SummaryResponse:
    try:
        df = data_manager.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    describe_df = df.describe(include="all", datetime_is_numeric=True).transpose()
    summary: Dict[str, Dict[str, Any]] = describe_df.replace({np.nan: None}).to_dict(
        orient="index"
    )
    return schemas.SummaryResponse(summary=summary)


@router.get("/{dataset_id}/columns")
def dataset_columns(dataset_id: str) -> Dict[str, Any]:
    try:
        metadata = data_manager.get_metadata(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    df = data_manager.get_dataset(dataset_id)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return {"columns": metadata.columns, "dtypes": dtypes}
