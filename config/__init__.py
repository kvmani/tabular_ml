"""Configuration loader utilities."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy available in runtime
    np = None  # type: ignore

from .schema import DatasetEntry, DatasetRegistry, Settings

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
BASE_CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "config.local.yaml"
ENV_PREFIX = "TABULAR_ML__"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_env_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path_parts = key[len(ENV_PREFIX) :].lower().split("__")
        target = overrides
        for part in path_parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        target[path_parts[-1]] = parsed
    return overrides


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    base = _load_yaml(BASE_CONFIG_PATH)
    local = _load_yaml(LOCAL_CONFIG_PATH)
    merged = _merge_dicts(base, local)
    env_overrides = _load_env_overrides()
    merged = _merge_dicts(merged, env_overrides)
    settings = Settings.model_validate(merged)

    random.seed(settings.app.random_seed)
    if np is not None:
        np.random.seed(settings.app.random_seed)
    os.environ.setdefault("PYTHONHASHSEED", str(settings.app.random_seed))
    return settings


@lru_cache(maxsize=1)
def load_dataset_registry(settings: Settings | None = None) -> DatasetRegistry:
    cfg = settings or load_settings()
    registry_path = Path(cfg.datasets.registry_file)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry file not found at {registry_path}"  # pragma: no cover - configuration issue
        )
    with registry_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(
            "Dataset registry must be a mapping of dataset keys to entries"
        )
    registry: DatasetRegistry = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Registry entry for {key} must be a mapping")
        registry[key] = DatasetEntry.model_validate(value)
    return registry


settings = load_settings()
