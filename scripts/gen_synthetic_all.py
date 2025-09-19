"""Generate synthetic expansions for all registry datasets (in-memory)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_dataset_registry, settings

from backend.app.core import config as runtime_config
from backend.app.services.data_augment import augment_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dataframe(filename: str) -> pd.DataFrame:
    path = (runtime_config.DATA_DIR / filename).resolve()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main() -> None:
    registry = load_dataset_registry()
    for key, entry in registry.items():
        df = load_dataframe(entry.file)
        augmented, info = augment_dataset(
            df,
            target_column=entry.target,
            task_type=entry.task,
            config=settings.datasets.synthetic,
        )
        print(
            f"Dataset {key} -> augmented rows: {info['final_rows']} (base {info['base_rows']}) "
            f"generators={info['applied_generators']}"
        )


if __name__ == "__main__":
    main()
