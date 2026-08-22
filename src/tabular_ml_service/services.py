"""Application orchestration shared by API routes and future CLI adapters."""

from __future__ import annotations

import json
import secrets
from io import BytesIO
from typing import Any

import joblib
from flask import current_app
from werkzeug.utils import secure_filename

from .config import Settings
from .core.data import parse_dataset, preview_records
from .core.outliers import OutlierConfig, OutlierResult, apply_outliers, detect_outliers
from .core.preprocessing import PreparationConfig, feature_plan
from .core.profiling import profile_dataset
from .core.training import FittedRun, TrainingOptions, predict_records, train_model
from .core.visualization import visualization_payload
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
    documented_target = workspace.metadata.get("target")
    documented_task = workspace.metadata.get("task")
    if (
        isinstance(documented_target, str)
        and documented_target in workspace.working.columns
        and documented_task in {"classification", "regression"}
    ):
        candidates = [
            item for item in profile["target_candidates"] if item["column"] != documented_target
        ]
        profile["target_candidates"] = [
            {
                "column": documented_target,
                "task": documented_task,
                "confidence": 100,
                "reason": "Documented target in the built-in dataset metadata",
            },
            *candidates,
        ]
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


def _mapping(payload: Any, name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ApiError(code="training.invalid_request", message=f"{name} must be an object.")
    return payload


def train_workspace(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    workspace = _store().get(session_id)
    target = payload.get("target")
    algorithm = payload.get("algorithm")
    if not isinstance(target, str) or not target:
        raise ApiError(code="target.missing", message="Choose a target column.")
    if not isinstance(algorithm, str) or not algorithm:
        raise ApiError(code="algorithm.missing", message="Choose an algorithm.")
    preparation = PreparationConfig.from_mapping(
        _mapping(payload.get("preparation"), "Preparation config")
    )
    options_payload = _mapping(payload.get("options"), "Training options")
    options_payload.setdefault("random_seed", _settings().random_seed)
    options = TrainingOptions.from_mapping(options_payload)
    run = train_model(
        workspace.working,
        target=target,
        algorithm_id=algorithm,
        parameters=_mapping(payload.get("parameters"), "Algorithm parameters"),
        preparation=preparation,
        options=options,
    )
    run_id = secrets.token_urlsafe(18)
    workspace.runs[run_id] = run
    return {"run_id": run_id, **run.result}


def _get_run(workspace: DatasetWorkspace, run_id: str) -> FittedRun:
    run = workspace.runs.get(run_id)
    if not isinstance(run, FittedRun):
        raise ApiError(
            code="run.not_found",
            message="This model run does not exist in the current workspace.",
            status=404,
        )
    return run


def get_run_report(session_id: str, run_id: str) -> dict[str, Any]:
    workspace = _store().get(session_id)
    run = _get_run(workspace, run_id)
    return {"run_id": run_id, **run.result}


def infer_workspace(
    session_id: str,
    run_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    workspace = _store().get(session_id)
    run = _get_run(workspace, run_id)
    return {"run_id": run_id, **predict_records(run, records)}


def compare_workspace(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    algorithms = payload.get("algorithms")
    if (
        not isinstance(algorithms, list)
        or not all(isinstance(item, str) for item in algorithms)
        or not 2 <= len(algorithms) <= 6
    ):
        raise ApiError(
            code="comparison.invalid_algorithms",
            message="Choose between two and six algorithms to compare.",
        )
    parameters = _mapping(payload.get("parameters"), "Comparison parameters")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for algorithm in algorithms:
        request_payload = {
            **payload,
            "algorithm": algorithm,
            "parameters": _mapping(parameters.get(algorithm), f"{algorithm} parameters"),
        }
        try:
            trained = train_workspace(session_id, request_payload)
        except ApiError as exc:
            failures.append(
                {
                    "algorithm": algorithm,
                    "error": {"code": exc.code, "message": exc.message},
                }
            )
            continue
        primary = trained["primary_metric"]
        results.append(
            {
                "run_id": trained["run_id"],
                "algorithm": trained["algorithm"],
                "primary_metric": primary,
                "test_score": trained["metrics"][primary],
                "cv_score": trained["cross_validation"]["metrics"][primary]["mean"],
                "metrics": trained["metrics"],
            }
        )
    results.sort(key=lambda item: float(item["test_score"]), reverse=True)
    if not results:
        raise ApiError(
            code="comparison.all_failed",
            message="None of the selected algorithms could be trained.",
            details={"failures": failures},
        )
    return {"leaderboard": results, "failures": failures}


def model_artifact(session_id: str, run_id: str) -> tuple[BytesIO, str]:
    workspace = _store().get(session_id)
    run = _get_run(workspace, run_id)
    artifact = {
        "format": "tabular-ml-joblib-v1",
        "notice": (
            "Generated by Tabular ML Workbench. Load only artifacts you created "
            "and trust; Python joblib files are executable deserialization formats."
        ),
        "target": run.target,
        "task": run.task,
        "algorithm": run.algorithm.id,
        "parameters": run.parameters,
        "feature_columns": run.feature_columns,
        "pipeline": run.pipeline,
        "report": run.result,
    }
    buffer = BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    buffer.seek(0)
    return buffer, f"tabular-ml-{run.algorithm.id}-{run_id[:8]}.joblib"


def visualize_workspace(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    workspace = _store().get(session_id)
    return visualization_payload(workspace.working, payload)
