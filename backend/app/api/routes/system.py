"""System and health endpoints."""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from config import load_dataset_registry, settings

from backend.app.core import config as runtime_config
from backend.app.log_stream import log_stream_manager
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


def _format_sse(data: str) -> str:
    return f"event: log\ndata: {data}\n\n"


@router.get("/system/logs/stream")
async def system_log_stream(level: str = Query("INFO", description="Minimum severity to stream")) -> StreamingResponse:
    """Stream structured application logs via server-sent events."""

    requested = level.upper()
    min_level = logging.INFO
    if requested == "DEBUG":
        min_level = logging.DEBUG
    elif requested not in {"INFO", "WARNING", "ERROR"}:
        raise HTTPException(status_code=400, detail="Unsupported log level")

    queue = log_stream_manager.register()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                event = await queue.get()
                if event.levelno < min_level:
                    continue
                yield _format_sse(event.to_json())
        finally:
            log_stream_manager.unregister(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
