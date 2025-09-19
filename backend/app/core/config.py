"""Runtime configuration helpers bridging to global settings."""
from __future__ import annotations

from pathlib import Path

from config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (PROJECT_ROOT / settings.datasets.root_dir).resolve()
SAMPLE_DATA_DIR = DATA_DIR
STORAGE_DIR = (PROJECT_ROOT / "backend" / "storage").resolve()


def ensure_directories() -> None:
    for path in {DATA_DIR, SAMPLE_DATA_DIR, STORAGE_DIR}:
        path.mkdir(parents=True, exist_ok=True)


def runtime_paths() -> dict:
    return {
        "data_dir": str(DATA_DIR),
        "sample_data_dir": str(SAMPLE_DATA_DIR),
        "storage_dir": str(STORAGE_DIR),
    }


def as_dict() -> dict:
    """Expose serialisable subset of runtime configuration for APIs."""
    return {
        "app": settings.app.model_dump(),
        "security": settings.security.model_dump(),
        "ml": settings.ml.model_dump(),
        "datasets": settings.datasets.model_dump(),
        "limits": settings.limits.model_dump(),
        "paths": runtime_paths(),
    }


ensure_directories()
