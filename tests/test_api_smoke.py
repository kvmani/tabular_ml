from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from config import settings
from backend.app.services.model_training import model_trainer


client = TestClient(app)


def _csrf_token():
    response = client.get("/health")
    token = response.headers["X-CSRF-Token"]
    return token, response


def test_health_and_algorithms():
    csrf_token, response = _csrf_token()
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

    sample_resp = client.post(
        "/data/samples/titanic",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert sample_resp.status_code == 200
    dataset = sample_resp.json()["dataset"]
    assert dataset["num_rows"] >= 1000
    dataset_id = dataset["dataset_id"]

    preview = client.get(f"/data/{dataset_id}/preview")
    assert preview.status_code == 200

    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "Survived",
        "task_type": "classification",
        "algorithm": "logistic_regression",
    }
    train_resp = client.post(
        "/model/train",
        json=train_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert train_resp.status_code == 200
    body = train_resp.json()
    assert "model_id" in body

    runs_resp = client.get("/runs/last")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["run_id"] == body["model_id"]

    if not settings.app.allow_file_uploads:
        upload_resp = client.post(
            "/data/upload",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert upload_resp.status_code in {404, 405}


def test_sample_flow_and_runs_endpoint():
    csrf_token, _ = _csrf_token()
    sample_resp = client.post(
        "/data/samples/titanic",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert sample_resp.status_code == 200
    dataset_id = sample_resp.json()["dataset"]["dataset_id"]

    split_payload = {
        "target_column": "Survived",
        "task_type": "classification",
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "stratify": True,
    }
    split_resp = client.post(
        f"/preprocess/{dataset_id}/split",
        json=split_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert split_resp.status_code == 200
    split_id = split_resp.json()["split_id"]

    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "Survived",
        "task_type": "classification",
        "algorithm": "logistic_regression",
        "split_id": split_id,
    }
    train_resp = client.post(
        "/model/train",
        json=train_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert train_resp.status_code == 200
    body = train_resp.json()
    assert "model_id" in body

    eval_resp = client.post(
        "/model/evaluate",
        json={"model_id": body["model_id"]},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert eval_resp.status_code == 200
    payload = eval_resp.json()
    assert "metrics" in payload
    assert payload["metrics"]

    runs_resp = client.get("/runs/last")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["run_id"] == body["model_id"]
