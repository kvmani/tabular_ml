"""Versioned HTTP API for Tabular ML."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, current_app, request, send_file

from ..core.algorithms import algorithm_catalog
from ..dataset_registry import DatasetRegistry
from ..errors import ApiError
from ..responses import failure, success
from ..services import (
    apply_outlier_treatment,
    compare_workspace,
    delete_workspace,
    get_run_report,
    get_workspace_payload,
    infer_workspace,
    load_builtin,
    load_upload,
    model_artifact,
    outlier_report,
    preparation_plan,
    reset_workspace,
    train_workspace,
    visualize_workspace,
)
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


@api_bp.get("/algorithms")
def algorithms() -> tuple[Response, int]:
    task = request.args.get("task")
    if task not in {None, "classification", "regression"}:
        raise ApiError(
            code="algorithm.invalid_task",
            message="Task must be classification or regression.",
        )
    return success({"algorithms": algorithm_catalog(task)})  # type: ignore[arg-type]


@api_bp.get("/datasets")
def datasets() -> tuple[Response, int]:
    return success({"datasets": DatasetRegistry().summaries()})


@api_bp.get("/datasets/<string:key>")
def datasets_metadata(key: str) -> tuple[Response, int]:
    try:
        metadata = DatasetRegistry().metadata(key)
    except KeyError as exc:
        raise ApiError(
            code="dataset.not_found",
            message=f"No built-in dataset named '{key}' exists.",
            status=404,
        ) from exc
    return success(metadata)


@api_bp.post("/datasets/<string:key>/load")
def datasets_load_builtin(key: str) -> tuple[Response, int]:
    return success(load_builtin(key), status=201)


@api_bp.post("/datasets/upload")
def datasets_upload() -> tuple[Response, int]:
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ApiError(
            code="upload.missing",
            message="Choose a CSV or Excel .xlsx file to continue.",
        )
    settings = current_app.extensions["tabular_ml_settings"]
    raw = upload.stream.read(settings.max_upload_bytes + 1)
    return success(load_upload(raw, upload.filename), status=201)


@api_bp.get("/sessions/<string:session_id>")
def sessions_get(session_id: str) -> tuple[Response, int]:
    return success(get_workspace_payload(session_id))


@api_bp.delete("/sessions/<string:session_id>")
def sessions_delete(session_id: str) -> tuple[Response, int]:
    return success(delete_workspace(session_id))


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(code="request.invalid_json", message="Send a JSON object.")
    return payload


@api_bp.post("/sessions/<string:session_id>/preparation/plan")
def preparation_get_plan(session_id: str) -> tuple[Response, int]:
    payload = _json_object()
    target = payload.get("target")
    if not isinstance(target, str) or not target:
        raise ApiError(code="target.missing", message="Choose a target column.")
    config_payload = payload.get("config")
    if config_payload is not None and not isinstance(config_payload, dict):
        raise ApiError(code="preparation.invalid", message="Preparation config must be an object.")
    return success(preparation_plan(session_id, target, config_payload))


@api_bp.post("/sessions/<string:session_id>/outliers/detect")
def outliers_detect(session_id: str) -> tuple[Response, int]:
    payload = _json_object()
    config_payload = payload.get("config")
    if config_payload is not None and not isinstance(config_payload, dict):
        raise ApiError(code="outliers.invalid", message="Outlier config must be an object.")
    target = payload.get("target")
    if target is not None and not isinstance(target, str):
        raise ApiError(code="target.invalid", message="Target must be a column name.")
    return success(outlier_report(session_id, config_payload, target))


@api_bp.post("/sessions/<string:session_id>/outliers/apply")
def outliers_apply(session_id: str) -> tuple[Response, int]:
    payload = _json_object()
    config_payload = payload.get("config")
    if config_payload is not None and not isinstance(config_payload, dict):
        raise ApiError(code="outliers.invalid", message="Outlier config must be an object.")
    action = payload.get("action")
    if not isinstance(action, str):
        raise ApiError(code="outliers.invalid_action", message="Choose an outlier action.")
    target = payload.get("target")
    if target is not None and not isinstance(target, str):
        raise ApiError(code="target.invalid", message="Target must be a column name.")
    return success(apply_outlier_treatment(session_id, config_payload, action, target))


@api_bp.post("/sessions/<string:session_id>/reset")
def sessions_reset(session_id: str) -> tuple[Response, int]:
    return success(reset_workspace(session_id))


@api_bp.post("/sessions/<string:session_id>/visualizations")
def visualizations_create(session_id: str) -> tuple[Response, int]:
    return success(visualize_workspace(session_id, _json_object()))


@api_bp.post("/sessions/<string:session_id>/runs")
def runs_train(session_id: str) -> tuple[Response, int]:
    return success(train_workspace(session_id, _json_object()), status=201)


@api_bp.post("/sessions/<string:session_id>/comparisons")
def runs_compare(session_id: str) -> tuple[Response, int]:
    return success(compare_workspace(session_id, _json_object()), status=201)


@api_bp.get("/sessions/<string:session_id>/runs/<string:run_id>")
def runs_get(session_id: str, run_id: str) -> tuple[Response, int]:
    return success(get_run_report(session_id, run_id))


@api_bp.post("/sessions/<string:session_id>/runs/<string:run_id>/infer")
def runs_infer(session_id: str, run_id: str) -> tuple[Response, int]:
    payload = _json_object()
    records = payload.get("rows")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ApiError(
            code="inference.invalid_rows",
            message="Prediction rows must be a list of objects.",
        )
    return success(infer_workspace(session_id, run_id, records))


@api_bp.get("/sessions/<string:session_id>/runs/<string:run_id>/artifact")
def runs_artifact(session_id: str, run_id: str) -> Response:
    buffer, filename = model_artifact(session_id, run_id)
    response = send_file(
        buffer,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=filename,
        max_age=0,
    )
    response.headers["X-Model-Format"] = "tabular-ml-joblib-v1"
    return response


@api_bp.errorhandler(ApiError)
def handle_api_error(error: ApiError) -> tuple[Response, int]:
    return failure(error)
