"""Validated discovery of tracked demonstration datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    key: str
    name: str
    task: str
    target: str
    file: str
    metadata_file: str
    rows: int
    columns: int
    summary: str
    license_name: str
    tags: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> DatasetRecord:
        required = {
            "key",
            "name",
            "task",
            "target",
            "file",
            "metadata_file",
            "rows",
            "columns",
            "summary",
            "license_name",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Dataset record is missing: {', '.join(missing)}")
        task = str(payload["task"])
        if task not in {"classification", "regression"}:
            raise ValueError(f"Unsupported dataset task: {task}")
        return cls(
            key=str(payload["key"]),
            name=str(payload["name"]),
            task=task,
            target=str(payload["target"]),
            file=str(payload["file"]),
            metadata_file=str(payload["metadata_file"]),
            rows=int(payload["rows"]),
            columns=int(payload["columns"]),
            summary=str(payload["summary"]),
            license_name=str(payload["license_name"]),
            tags=tuple(str(tag) for tag in payload.get("tags", [])),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "task": self.task,
            "target": self.target,
            "rows": self.rows,
            "columns": self.columns,
            "summary": self.summary,
            "license": self.license_name,
            "tags": list(self.tags),
        }


class DatasetRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(str(files("tabular_ml_service").joinpath("datasets")))
        raw = json.loads((self.root / "registry.json").read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or not isinstance(raw.get("datasets"), list):
            raise ValueError("Unsupported dataset registry schema")
        records = [DatasetRecord.from_mapping(item) for item in raw["datasets"]]
        if len({record.key for record in records}) != len(records):
            raise ValueError("Dataset registry keys must be unique")
        self._records = {record.key: record for record in records}

    def summaries(self) -> list[dict[str, Any]]:
        return [record.public_dict() for record in self._records.values()]

    def get(self, key: str) -> DatasetRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"Unknown built-in dataset: {key}") from exc

    def metadata(self, key: str) -> dict[str, Any]:
        record = self.get(key)
        payload = json.loads((self.root / record.metadata_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset metadata must be an object: {key}")
        return payload

    def validate_files(self) -> list[str]:
        issues: list[str] = []
        for record in self._records.values():
            if not (self.root / record.file).is_file():
                issues.append(f"{record.key}: missing CSV {record.file}")
            if not (self.root / record.metadata_file).is_file():
                issues.append(f"{record.key}: missing metadata {record.metadata_file}")
        return issues
