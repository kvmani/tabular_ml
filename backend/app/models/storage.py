"""Data storage models for in-memory runtime objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata describing a dataset stored in memory."""

    dataset_id: str
    name: str
    source: str
    description: Optional[str]
    created_at: datetime
    num_rows: int
    num_columns: int
    columns: List[str]
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSplit:
    """Container for data splits used during model training."""

    split_id: str
    dataset_id: str
    feature_columns: List[str]
    target_column: str
    task_type: str
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelArtifact:
    """Encapsulates a trained model and its metadata."""

    model_id: str
    algorithm: str
    task_type: str
    target_column: str
    feature_columns: List[str]
    model_object: Any
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    split_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
