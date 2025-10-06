from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.data_manager import data_manager


client = TestClient(app)


def test_datasets_endpoint_returns_default_dataset():
    data_manager.ensure_default_dataset()

    response = client.get("/data/datasets")
    assert response.status_code == 200

    payload = response.json()
    assert payload["datasets"], "datasets list should not be empty"

    default_id = payload.get("default_dataset_id")
    assert default_id, "default_dataset_id should be set"

    dataset_ids = {dataset["dataset_id"] for dataset in payload["datasets"]}
    assert default_id in dataset_ids
