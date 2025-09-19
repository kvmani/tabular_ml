"""Pydantic schemas for API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset schemas
# ---------------------------------------------------------------------------


class DatasetMetadataModel(BaseModel):
    dataset_id: str
    name: str
    source: str
    description: Optional[str]
    created_at: datetime
    num_rows: int
    num_columns: int
    columns: List[str]


class DatasetListResponse(BaseModel):
    datasets: List[DatasetMetadataModel]


class UploadResponse(BaseModel):
    dataset: DatasetMetadataModel


class SampleDatasetModel(BaseModel):
    key: str
    name: str
    description: str
    target_column: str


class SampleDatasetListResponse(BaseModel):
    samples: List[SampleDatasetModel]


class PreviewResponse(BaseModel):
    data: List[Dict[str, Any]]


class SummaryResponse(BaseModel):
    summary: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class OutlierDetectionRequest(BaseModel):
    columns: Optional[List[str]] = None
    z_threshold: float = Field(3.0, gt=0)


class OutlierDetectionResponse(BaseModel):
    total_outliers: int
    inspected_columns: List[str]
    sample_indices: List[int]


class OutlierRemovalResponse(BaseModel):
    dataset: DatasetMetadataModel
    report: OutlierDetectionResponse


class ImputationRequest(BaseModel):
    strategy: str = Field("mean", description="Imputation strategy such as mean/median/most_frequent")
    columns: Optional[List[str]] = None
    fill_value: Optional[float] = None


class ImputationResponse(BaseModel):
    dataset: DatasetMetadataModel


class FilterRule(BaseModel):
    column: str
    operator: str = Field("eq", description="Comparison operator (eq, ne, gt, gte, lt, lte, contains, in)")
    value: Any


class FilterRequest(BaseModel):
    rules: List[FilterRule]


class FilterResponse(BaseModel):
    dataset: DatasetMetadataModel


class SplitRequest(BaseModel):
    target_column: str
    task_type: str = Field(..., regex="^(classification|regression)$")
    test_size: float = Field(0.2, gt=0, lt=0.8)
    val_size: float = Field(0.2, gt=0, lt=0.8)
    random_state: int = 42
    stratify: bool = True


class SplitResponse(BaseModel):
    split_id: str
    dataset_id: str
    target_column: str
    feature_columns: List[str]


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


class AlgorithmModel(BaseModel):
    key: str
    label: str
    task_types: List[str]


class AlgorithmListResponse(BaseModel):
    algorithms: List[AlgorithmModel]


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str = Field(..., regex="^(classification|regression)$")
    algorithm: str
    split_id: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TrainingResponse(BaseModel):
    model_id: str
    metrics: Dict[str, Dict[str, Any]]
    history: List[Dict[str, Any]]
    split_id: str


class ModelSummary(BaseModel):
    model_id: str
    algorithm: str
    task_type: str
    target_column: str
    metrics: Dict[str, Any]
    created_at: datetime


class ModelListResponse(BaseModel):
    models: List[ModelSummary]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    model_id: str


class EvaluationResponse(BaseModel):
    metrics: Dict[str, Any]
    confusion_matrix: Optional[Dict[str, Any]] = None
    roc_curve: Optional[Dict[str, Any]] = None
    regression_diagnostics: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class HistogramRequest(BaseModel):
    column: str


class ScatterRequest(BaseModel):
    x: str
    y: str
    color: Optional[str] = None
