"""Registry of built-in sample datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from config import load_dataset_registry

from backend.app.core.config import DATA_DIR


@dataclass(frozen=True)
class SampleDataset:
    """Metadata describing a curated dataset available to users."""

    key: str
    name: str
    description: str
    target_column: str
    task: str
    path: Path


def load_registry() -> Dict[str, SampleDataset]:
    """Return a dictionary of available sample datasets keyed by identifier."""

    registry = load_dataset_registry()
    datasets: Dict[str, SampleDataset] = {}
    for key, entry in registry.items():
        path = (DATA_DIR / entry.file).resolve()
        datasets[key] = SampleDataset(
            key=key,
            name=entry.name or key.replace("_", " ").title(),
            description=entry.description or "Synthetic offline dataset",
            target_column=entry.target,
            task=entry.task,
            path=path,
        )
    return datasets


SAMPLE_DATASETS = load_registry()
