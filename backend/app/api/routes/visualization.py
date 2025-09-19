"""Visualization API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api import schemas
from backend.app.services import visualization as viz
from backend.app.services.data_manager import data_manager

router = APIRouter(prefix="/visualization", tags=["visualization"])


def _fetch_dataset(dataset_id: str):
    try:
        return data_manager.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{dataset_id}/histogram")
def histogram(dataset_id: str, request: schemas.HistogramRequest) -> dict:
    df = _fetch_dataset(dataset_id)
    if request.column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found in dataset")
    figure = viz.histogram(df, request.column)
    return {"figure": figure}


@router.post("/{dataset_id}/scatter")
def scatter(dataset_id: str, request: schemas.ScatterRequest) -> dict:
    df = _fetch_dataset(dataset_id)
    for column in (
        (request.x, request.y, request.color)
        if request.color
        else (request.x, request.y)
    ):
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
    figure = viz.scatter(df, request.x, request.y, color=request.color)
    return {"figure": figure}
