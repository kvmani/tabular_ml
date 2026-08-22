from __future__ import annotations

import io


def _session(client) -> tuple[str, list[dict]]:
    rows = ["record_id,signal,group,target"]
    records = []
    for index in range(80):
        signal = (index - 40) / 10
        group = "lab" if index % 3 == 0 else "plant"
        target = int(signal + (0.8 if group == "lab" else 0) > 0)
        rows.append(f"R-{index},{signal},{group},{target}")
        records.append({"record_id": f"R-{index}", "signal": signal, "group": group})
    response = client.post(
        "/tabular_ml/api/v1/datasets/upload",
        data={"file": (io.BytesIO(("\n".join(rows) + "\n").encode()), "training.csv")},
        content_type="multipart/form-data",
    )
    return response.get_json()["data"]["session_id"], records


def _train_payload(algorithm: str = "linear") -> dict:
    return {
        "target": "target",
        "algorithm": algorithm,
        "parameters": {},
        "preparation": {"scaling": "standard"},
        "options": {
            "test_size": 0.25,
            "cv_folds": 2,
            "random_seed": 11,
            "permutation_importance": False,
        },
    }


def test_algorithm_catalog_training_inference_report_and_artifact(client) -> None:
    catalog = client.get("/tabular_ml/api/v1/algorithms?task=classification")
    ids = {item["id"] for item in catalog.get_json()["data"]["algorithms"]}
    assert {"random_forest", "gradient_boosting", "knn"} <= ids

    session_id, records = _session(client)
    trained = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/runs",
        json=_train_payload(),
    )
    data = trained.get_json()["data"]
    assert trained.status_code == 201
    assert data["metrics"]["balanced_accuracy"] >= 0.8
    run_id = data["run_id"]

    report = client.get(f"/tabular_ml/api/v1/sessions/{session_id}/runs/{run_id}")
    assert report.get_json()["data"]["run_id"] == run_id

    inferred = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/runs/{run_id}/infer",
        json={"rows": records[:3]},
    )
    assert len(inferred.get_json()["data"]["rows"]) == 3

    artifact = client.get(f"/tabular_ml/api/v1/sessions/{session_id}/runs/{run_id}/artifact")
    assert artifact.status_code == 200
    assert artifact.headers["X-Model-Format"] == "tabular-ml-joblib-v1"
    assert artifact.data.startswith(b"x")


def test_model_comparison_creates_ranked_runs(client) -> None:
    session_id, _ = _session(client)
    payload = _train_payload()
    payload.pop("algorithm")
    payload["algorithms"] = ["linear", "decision_tree"]
    response = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/comparisons",
        json=payload,
    )
    data = response.get_json()["data"]
    assert response.status_code == 201
    assert len(data["leaderboard"]) == 2
    assert data["leaderboard"][0]["test_score"] >= data["leaderboard"][1]["test_score"]


def test_training_routes_return_stable_errors(client) -> None:
    session_id, _ = _session(client)
    invalid = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/runs",
        json={"target": "target"},
    )
    missing_run = client.get(f"/tabular_ml/api/v1/sessions/{session_id}/runs/not-real")
    assert invalid.get_json()["error"]["code"] == "algorithm.missing"
    assert missing_run.status_code == 404
    assert missing_run.get_json()["error"]["code"] == "run.not_found"
