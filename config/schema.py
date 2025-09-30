"""Pydantic schemas describing configuration files."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict


class AppCfg(BaseModel):
    name: str
    environment: str
    host: str
    port: int = Field(ge=1, le=65535)
    debug: bool
    allow_file_uploads: bool
    default_sample_dataset: Optional[str] = None
    max_upload_mb: int = Field(gt=0)
    random_seed: int
    log_level: str
    persist_user_data: bool

    @field_validator("environment")
    @classmethod
    def _lower_env(cls, value: str) -> str:
        return value.lower()


class SecurityCfg(BaseModel):
    csp_enabled: bool
    cors_origins: List[str]
    csrf_protect: bool
    csp_policy: str


class MLCfg(BaseModel):
    default_task: str
    default_algorithms: List[str]
    sklearn_onehot_sparse: bool
    n_jobs: int
    timeout_sec: int = Field(gt=0)


class DatasetEntry(BaseModel):
    file: str
    target: str
    task: str
    name: Optional[str] = None
    description: Optional[str] = None


class SyntheticCfg(BaseModel):
    enable: bool
    per_real_row_multiplier: int = Field(default=1, ge=1)
    generators: List[str]

    @field_validator("generators")
    @classmethod
    def _ensure_unique(cls, value: List[str]) -> List[str]:
        return list(dict.fromkeys(value))


class DatasetsCfg(BaseModel):
    root_dir: str
    registry_file: str
    synthetic: SyntheticCfg


class LimitsCfg(BaseModel):
    max_rows_preview: int = Field(gt=0)
    max_rows_train: int = Field(gt=0)
    max_cols: int = Field(gt=0)


class Settings(BaseModel):
    app: AppCfg
    security: SecurityCfg
    ml: MLCfg
    datasets: DatasetsCfg
    limits: LimitsCfg

    model_config = ConfigDict(frozen=True)


DatasetRegistry = Dict[str, DatasetEntry]


