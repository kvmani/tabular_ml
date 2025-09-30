"""Data management utilities for the ML platform."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

import pandas as pd

from config import settings

from backend.app.models.storage import DataSplit, DatasetMetadata, ModelArtifact
from backend.app.services.data_augment import augment_dataset
from backend.app.utils.dataset_registry import SAMPLE_DATASETS


class DataManager:
    """Singleton-style in-memory storage for datasets, splits, and models."""

    def __init__(self) -> None:
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, DatasetMetadata] = {}
        self._splits: Dict[str, DataSplit] = {}
        self._models: Dict[str, ModelArtifact] = {}
        self._lock = Lock()
        self.sample_datasets = SAMPLE_DATASETS
        self.synthetic_cfg = settings.datasets.synthetic
        self.default_dataset_id: Optional[str] = None

        self._preload_default_dataset()

    def _preload_default_dataset(self) -> None:
        """Load the configured default sample dataset if one is provided."""

        default_key = getattr(settings.app, "default_sample_dataset", None)
        if not default_key:
            return
        if default_key not in self.sample_datasets:
            logging.getLogger(__name__).warning(
                "Default sample dataset '%s' is not present in the registry.",
                default_key,
            )
            return
        try:
            metadata = self.load_sample_dataset(default_key)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).warning(
                "Failed to preload default dataset '%s': %s", default_key, exc
            )
            return
        with self._lock:
            if metadata.dataset_id in self._datasets:
                self.default_dataset_id = metadata.dataset_id

    def ensure_default_dataset(self) -> None:
        """Ensure the default dataset is available after state resets."""

        default_key = getattr(settings.app, "default_sample_dataset", None)
        if not default_key:
            return
        with self._lock:
            if self.default_dataset_id and self.default_dataset_id in self._datasets:
                return
            self.default_dataset_id = None
        self._preload_default_dataset()

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------
    def create_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        source: str,
        description: Optional[str] = None,
    ) -> DatasetMetadata:
        """Store a dataframe and return its metadata."""

        if df.shape[1] > settings.limits.max_cols:
            raise ValueError(
                f"Dataset has {df.shape[1]} columns which exceeds the limit of {settings.limits.max_cols}."
            )
        dataset_id = uuid4().hex
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            source=source,
            description=description,
            created_at=datetime.now(timezone.utc),
            num_rows=int(df.shape[0]),
            num_columns=int(df.shape[1]),
            columns=list(df.columns),
        )
        with self._lock:
            self._datasets[dataset_id] = df
            self._metadata[dataset_id] = metadata
        return metadata

    def list_datasets(self) -> List[Dict[str, object]]:
        """Return metadata for all stored datasets."""

        self.ensure_default_dataset()
        with self._lock:
            return [asdict(meta) for meta in self._metadata.values()]

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        try:
            return self._datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Dataset {dataset_id} not found") from exc

    def get_metadata(self, dataset_id: str) -> DatasetMetadata:
        try:
            return self._metadata[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Metadata for dataset {dataset_id} not found") from exc

    def clone_dataset(
        self, dataset_id: str, name_suffix: str, description: Optional[str] = None
    ) -> DatasetMetadata:
        """Create a copy of an existing dataset with optional new description."""

        df = self.get_dataset(dataset_id).copy()
        source = f"derived:{dataset_id}"
        metadata = self.create_dataset(
            df,
            name=f"{self.get_metadata(dataset_id).name} {name_suffix}",
            source=source,
            description=description,
        )
        return metadata

    # ------------------------------------------------------------------
    # Sample datasets
    # ------------------------------------------------------------------
    def list_sample_datasets(self) -> List[Dict[str, object]]:
        """Return metadata describing built-in datasets."""

        samples: List[Dict[str, object]] = []
        for key, dataset in self.sample_datasets.items():
            samples.append(
                {
                    "key": key,
                    "name": dataset.name,
                    "description": dataset.description,
                    "target_column": dataset.target_column,
                    "task": dataset.task,
                    "synthetic_available": self.synthetic_cfg.enable,
                }
            )
        return samples

    def load_sample_dataset(self, key: str) -> DatasetMetadata:
        if key not in self.sample_datasets:
            raise KeyError(f"Unknown sample dataset: {key}")
        dataset = self.sample_datasets[key]
        if not dataset.path.exists():
            raise FileNotFoundError(
                f"Sample dataset file missing: {dataset.path}. Please ensure datasets are bundled."
            )
        if dataset.path.suffix.lower() == ".csv":
            df = pd.read_csv(dataset.path)
        else:
            df = pd.read_excel(dataset.path)
        metadata = self.create_dataset(
            df=df,
            name=dataset.name,
            source=f"sample:{dataset.key}",
            description=dataset.description,
        )
        metadata.extras["task"] = dataset.task
        metadata.extras["target_column"] = dataset.target_column

        if self.synthetic_cfg.enable:
            augmented_df, augmentation_info = augment_dataset(
                df,
                target_column=dataset.target_column,
                task_type=dataset.task,
                config=self.synthetic_cfg,
            )
            augmented_metadata = self.create_dataset(
                df=augmented_df,
                name=f"{dataset.name} (Augmented)",
                source=f"sample:{dataset.key}:synthetic",
                description=f"{dataset.description} [Synthetic augmentation]",
            )
            augmented_metadata.extras.update(
                {
                    "augmentation": augmentation_info,
                    "target_column": dataset.target_column,
                    "task": dataset.task,
                    "origin_dataset_id": metadata.dataset_id,
                }
            )
            metadata.extras["augmented_dataset_id"] = augmented_metadata.dataset_id
        return metadata

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    def store_split(self, split: DataSplit) -> DataSplit:
        with self._lock:
            self._splits[split.split_id] = split
        return split

    def get_split(self, split_id: str) -> DataSplit:
        try:
            return self._splits[split_id]
        except KeyError as exc:
            raise KeyError(f"Split {split_id} not found") from exc

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def store_model(self, artifact: ModelArtifact) -> ModelArtifact:
        with self._lock:
            self._models[artifact.model_id] = artifact
        return artifact

    def get_model(self, model_id: str) -> ModelArtifact:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Model {model_id} not found") from exc

    def list_models(self) -> Iterable[Dict[str, object]]:
        with self._lock:
            return [
                {
                    "model_id": artifact.model_id,
                    "algorithm": artifact.algorithm,
                    "created_at": artifact.created_at.isoformat(),
                    "target_column": artifact.target_column,
                    "task_type": artifact.task_type,
                    "metrics": artifact.metrics,
                }
                for artifact in self._models.values()
            ]


data_manager = DataManager()
