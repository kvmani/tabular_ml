from __future__ import annotations

import json

import pytest

from tabular_ml_service.dataset_registry import DatasetRegistry


def test_empty_packaged_registry_is_valid() -> None:
    registry = DatasetRegistry()
    assert registry.summaries() == []
    assert registry.validate_files() == []


def test_registry_rejects_duplicate_keys(tmp_path) -> None:
    item = {
        "key": "demo",
        "name": "Demo",
        "task": "classification",
        "target": "label",
        "file": "demo.csv",
        "metadata_file": "demo.json",
        "rows": 2,
        "columns": 2,
        "summary": "Demo",
        "license_name": "CC0",
    }
    (tmp_path / "registry.json").write_text(
        json.dumps({"schema_version": 1, "datasets": [item, item]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        DatasetRegistry(tmp_path)
