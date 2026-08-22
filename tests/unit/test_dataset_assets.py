from __future__ import annotations

import hashlib
import json

import pandas as pd

from tabular_ml_service.dataset_registry import DatasetRegistry


def test_every_dataset_matches_machine_readable_metadata() -> None:
    registry = DatasetRegistry()
    for summary in registry.summaries():
        record = registry.get(summary["key"])
        csv_path = registry.root / record.file
        metadata = registry.metadata(record.key)
        frame = pd.read_csv(csv_path)
        assert len(frame) == metadata["rows"] == record.rows
        assert len(frame.columns) == metadata["columns"] == record.columns
        assert metadata["target"] in frame.columns
        assert metadata["column_names"] == frame.columns.tolist()
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == metadata["sha256"]
        assert metadata["license"]["name"]
        assert metadata["license"]["url"].startswith("https://")
        assert metadata["license"]["attribution"]
        assert len(metadata["learning_goals"]) >= 3
        assert metadata["ethical_notes"]


def test_registry_json_is_portable_and_has_unique_keys() -> None:
    registry = DatasetRegistry()
    payload = json.loads((registry.root / "registry.json").read_text(encoding="utf-8"))
    keys = [item["key"] for item in payload["datasets"]]
    assert len(keys) == len(set(keys)) == 6
