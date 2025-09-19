from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from config import settings
from backend.app.services.model_training import model_trainer


client = TestClient(app)


def test_health_and_algorithms():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["allow_file_uploads"] is settings.app.allow_file_uploads

    algos = client.get("/model/algorithms")
    assert algos.status_code == 200
    algo_keys = {a["key"] for a in algos.json()["algorithms"]}
    assert "random_forest" in algo_keys
    if model_trainer.torch_available:
        assert "neural_network" in algo_keys
    else:
        assert "neural_network" not in algo_keys


def test_sample_flow_and_runs_endpoint():
    sample_resp = client.post("/data/samples/titanic")
    assert sample_resp.status_code == 200
    dataset_id = sample_resp.json()["dataset"]["dataset_id"]

    preview = client.get(f"/data/{dataset_id}/preview")
    assert preview.status_code == 200

    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "Survived",
        "task_type": "classification",
        "algorithm": "logistic_regression",
    }
    train_resp = client.post("/model/train", json=train_payload)
    assert train_resp.status_code == 200
    body = train_resp.json()
    assert "model_id" in body

    runs_resp = client.get("/runs/last")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["run_id"] == body["model_id"]

    if not settings.app.allow_file_uploads:
        upload_resp = client.post("/data/upload")
        assert upload_resp.status_code in {404, 405}
