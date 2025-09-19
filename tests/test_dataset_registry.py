from __future__ import annotations

from config import load_dataset_registry, settings

from backend.app.services.data_manager import data_manager


def test_registry_contains_many_entries():
    registry = load_dataset_registry()
    assert len(registry) >= 10

    samples = data_manager.list_sample_datasets()
    assert len(samples) >= 10


def test_sample_load_creates_augmented_copy():
    metadata = data_manager.load_sample_dataset("titanic")
    stored_metadata = data_manager.get_metadata(metadata.dataset_id)

    assert stored_metadata.extras["target_column"] == "Survived"

    if settings.datasets.synthetic.enable:
        assert "augmented_dataset_id" in stored_metadata.extras
        augmented_id = stored_metadata.extras["augmented_dataset_id"]
        augmented_metadata = data_manager.get_metadata(augmented_id)
        assert augmented_metadata.extras["origin_dataset_id"] == metadata.dataset_id
        augmentation = augmented_metadata.extras.get("augmentation", {})
        assert augmentation.get("final_rows", 0) >= augmentation.get("target_rows", 0)

