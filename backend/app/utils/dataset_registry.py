"""Registry of built-in sample datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from backend.app.core.config import settings


@dataclass(frozen=True)
class SampleDataset:
    """Metadata describing a curated dataset available to users."""

    key: str
    name: str
    filename: str
    description: str
    target_column: str

    @property
    def path(self) -> Path:
        return settings.SAMPLE_DATA_DIR / self.filename


def load_registry() -> Dict[str, SampleDataset]:
    """Return a dictionary of available sample datasets keyed by identifier."""

    datasets: List[SampleDataset] = [
        SampleDataset(
            key="titanic",
            name="Titanic Survival",
            filename="titanic_sample.csv",
            description="Passenger information from the Titanic voyage with survival labels.",
            target_column="Survived",
        ),
        SampleDataset(
            key="adult_income",
            name="US Census Income",
            filename="adult_income_sample.csv",
            description="Census features for predicting whether annual income exceeds $50K.",
            target_column="income",
        ),
    ]
    return {dataset.key: dataset for dataset in datasets}


SAMPLE_DATASETS = load_registry()
