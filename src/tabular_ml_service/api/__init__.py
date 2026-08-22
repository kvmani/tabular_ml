"""Versioned HTTP API for Tabular ML."""

from __future__ import annotations

from flask import Blueprint, Response, current_app

from ..dataset_registry import DatasetRegistry
from ..errors import ApiError
from ..responses import failure, success
from ..version import __version__

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


@api_bp.get("/health")
def health() -> tuple[Response, int]:
    registry = DatasetRegistry()
    issues = registry.validate_files()
    state = "ok" if not issues else "degraded"
    return success(
        {
            "status": state,
            "service": "tabular_ml",
            "version": __version__,
            "dataset_registry_issues": issues,
        },
        status=200 if not issues else 503,
    )


@api_bp.get("/config")
def config() -> tuple[Response, int]:
    settings = current_app.extensions["tabular_ml_settings"]
    return success(
        {
            "version": __version__,
            "limits": settings.public_dict(),
            "privacy": {
                "persistence": "none",
                "telemetry": False,
                "runtime_network_calls": False,
                "session_idle_expiry_seconds": settings.session_ttl_seconds,
            },
        }
    )


@api_bp.get("/datasets")
def datasets() -> tuple[Response, int]:
    return success({"datasets": DatasetRegistry().summaries()})


@api_bp.errorhandler(ApiError)
def handle_api_error(error: ApiError) -> tuple[Response, int]:
    return failure(error)
