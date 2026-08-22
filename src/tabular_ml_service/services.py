"""Application orchestration shared by API routes and future CLI adapters."""

from __future__ import annotations

import json
from typing import Any

from flask import current_app
from werkzeug.utils import secure_filename

from .config import Settings
from .core.data import parse_dataset, preview_records
from .core.outliers import OutlierConfig, OutlierResult, apply_outliers, detect_outliers
from .core.preprocessing import PreparationConfig, feature_plan
from .core.profiling import profile_dataset
from .dataset_registry import DatasetRegistry
from .errors import ApiError
from .store import BoundedTTLStore
from .workspace import DatasetWorkspace


def _settings() -> Settings:
    value = current_app.extensions["tabular_ml_settings"]
    if not isinstance(value, Settings):
        raise RuntimeError("Tabular ML settings extension is invalid")
    return value


def _store() -> BoundedTTLStore[DatasetWorkspace]:
    value: BoundedTTLStore[DatasetWorkspace] = current_app.extensions["tabular_ml_store"]
    return value


def _workspace_payload(session_id: str, workspace: DatasetWorkspace) -> dict[str, Any]:
    settings = _settings()
    profile = profile_dataset(workspace.working)
    return {
        "session_id": session_id,
        "dataset": {
            "name": workspace.name,
            "source": workspace.source,
            "rows": len(workspace.working),
            "columns": len(workspace.working.columns),
            "metadata": workspace.metadata,
        },
        "profile": profile,
        "preview": {
            "columns": [str(column) for column in workspace.working.columns],
            "rows": preview_records(workspace.working, settings.preview_rows),
            "shown": min(len(workspace.working), settings.preview_rows),
        },
        "history": workspace.history,
    }


def load_upload(raw: bytes, filename: str) -> dict[str, Any]:
    safe_name = secure_filename(filename) or "uploaded.csv"
    frame = parse_dataset(raw, safe_name, _settings())
    workspace = DatasetWorkspace.create(name=safe_name, source="upload", frame=frame)
    session_id = _store().put(workspace)
    return _workspace_payload(session_id, workspace)


def load_builtin(key: str) -> dict[str, Any]:
    registry = DatasetRegistry()
    try:
        record = registry.get(key)
    except KeyError as exc:
        raise ApiError(
            code="dataset.not_found",
            message=f"No built-in dataset named '{key}' exists.",
            status=404,
        ) from exc
    raw = (registry.root / record.file).read_bytes()
    frame = parse_dataset(raw, record.file, _settings())
    metadata_path = registry.root / record.metadata_file
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    workspace = DatasetWorkspace.create(
        name=record.name,
        source=f"built-in:{record.key}",
        frame=frame,
        metadata=metadata,
    )
    session_id = _store().put(workspace)
    return _workspace_payload(session_id, workspace)


def get_workspace_payload(session_id: str) -> dict[str, Any]:
    return _workspace_payload(session_id, _store().get(session_id))


def delete_workspace(session_id: str) -> dict[str, Any]:
    deleted = _store().delete(session_id)
    if not deleted:
        raise ApiError(
            code="session.not_found",
            message="This workspace expired or does not exist.",
            status=404,
        )
    return {"deleted": True, "session_id": session_id}


def preparation_plan(
    session_id: str,
    target: str,
    config_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    workspace = _store().get(session_id)
    config = PreparationConfig.from_mapping(config_payload)
    return feature_plan(workspace.working, target, config)


def _outlier_result(
    workspace: DatasetWorkspace,
    config_payload: dict[str, Any] | None,
    target: str | None,
) -> OutlierResult:
    config = OutlierConfig.from_mapping(config_payload)
    detection_frame = workspace.working
    if target and not config.columns and target in detection_frame.columns:
        detection_frame = detection_frame.drop(columns=[target])
    return detect_outliers(
        detection_frame,
        config,
        random_seed=_settings().random_seed,
    )


def outlier_report(
    session_id: str,
    config_payload: dict[str, Any] | None,
    target: str | None = None,
) -> dict[str, Any]:
    workspace = _store().get(session_id)
    return _outlier_result(workspace, config_payload, target).public_dict()


def apply_outlier_treatment(
    session_id: str,
    config_payload: dict[str, Any] | None,
    action: str,
    target: str | None = None,
) -> dict[str, Any]:
    workspace = _store().get(session_id)
    result = _outlier_result(workspace, config_payload, target)
    before = len(workspace.working)
    workspace.working = apply_outliers(workspace.working, result, action)  # type: ignore[arg-type]
    workspace.history.append(
        {
            "operation": "outlier_treatment",
            "method": result.config.method,
            "action": action,
            "outlier_rows": int(result.row_mask.sum()),
            "rows_before": before,
            "rows_after": len(workspace.working),
        }
    )
    return _workspace_payload(session_id, workspace)


def reset_workspace(session_id: str) -> dict[str, Any]:
    workspace = _store().get(session_id)
    workspace.reset()
    return _workspace_payload(session_id, workspace)
