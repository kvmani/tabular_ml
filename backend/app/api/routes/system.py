"""System and health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from config import load_dataset_registry, settings

from backend.app.core import config as runtime_config
from backend.app.services.run_tracker import run_tracker

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app.name,
        "environment": settings.app.environment,
        "allow_file_uploads": settings.app.allow_file_uploads,
    }


@router.get("/system/config")
def system_config() -> dict:
    registry = load_dataset_registry(settings)
    return {
        "settings": runtime_config.as_dict(),
        "dataset_registry": {
            key: entry.model_dump() for key, entry in registry.items()
        },
    }


@router.get("/runs/last")
def last_run() -> dict:
    summary = run_tracker.get_last()
    if summary is None:
        raise HTTPException(status_code=404, detail="No runs recorded yet")
    return summary.to_dict()
