"""API helper utilities."""
from __future__ import annotations

from dataclasses import asdict

from backend.app.api import schemas
from backend.app.models.storage import DatasetMetadata


def metadata_to_model(metadata: DatasetMetadata) -> schemas.DatasetMetadataModel:
    if isinstance(metadata, schemas.DatasetMetadataModel):
        return metadata
    return schemas.DatasetMetadataModel.model_validate(asdict(metadata))
