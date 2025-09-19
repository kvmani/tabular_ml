from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def _csrf_token() -> str:
    response = client.get("/health")
    return response.headers["X-CSRF-Token"]


def test_offline_pipeline_end_to_end():
    csrf_token = _csrf_token()

    sample_resp = client.post(
        "/data/samples/titanic",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert sample_resp.status_code == 200
    dataset = sample_resp.json()["dataset"]
    assert dataset["num_rows"] >= 1000
    dataset_id = dataset["dataset_id"]

    preview_resp = client.get(f"/data/{dataset_id}/preview")
    summary_resp = client.get(f"/data/{dataset_id}/summary")
    columns_resp = client.get(f"/data/{dataset_id}/columns")

    assert preview_resp.status_code == 200
    assert summary_resp.status_code == 200
    assert columns_resp.status_code == 200
    assert columns_resp.json()["columns"]

    split_payload = {
        "target_column": "Survived",
        "task_type": "classification",
        "test_size": 0.25,
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
        "algorithm": "random_forest",
        "split_id": split_id,
    }
    train_resp = client.post(
        "/model/train",
        json=train_payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert train_resp.status_code == 200
    model_id = train_resp.json()["model_id"]

    eval_resp = client.post(
        "/model/evaluate",
        json={"model_id": model_id},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert eval_resp.status_code == 200
    metrics = eval_resp.json()["metrics"]
    assert "validation" in metrics
    assert "test" in metrics

    system_resp = client.get("/system/config")
    assert system_resp.status_code == 200
    system_payload = system_resp.json()
    assert "dataset_registry" in system_payload
    assert "settings" in system_payload

