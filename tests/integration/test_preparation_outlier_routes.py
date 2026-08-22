from __future__ import annotations

import io


def _session(client) -> str:
    rows = ["record_id,measurement,group,target"]
    rows.extend(f"R-{index},{index},g-{index % 3},{index % 2}" for index in range(30))
    rows.append("R-30,500,g-0,0")
    response = client.post(
        "/tabular_ml/api/v1/datasets/upload",
        data={"file": (io.BytesIO(("\n".join(rows) + "\n").encode()), "outliers.csv")},
        content_type="multipart/form-data",
    )
    return response.get_json()["data"]["session_id"]


def test_preparation_and_reversible_outlier_flow(client) -> None:
    session_id = _session(client)
    plan_response = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/preparation/plan",
        json={
            "target": "target",
            "config": {"numeric_imputer": "median", "scaling": "robust"},
        },
    )
    plan = plan_response.get_json()["data"]
    assert plan["task"] == "classification"
    assert plan["groups"]["numeric"] == ["measurement"]

    detected = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/outliers/detect",
        json={"target": "target", "config": {"method": "iqr", "threshold": 1.5}},
    )
    assert detected.get_json()["data"]["outlier_rows"] == 1

    applied = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/outliers/apply",
        json={
            "target": "target",
            "action": "drop",
            "config": {"method": "iqr", "threshold": 1.5},
        },
    )
    applied_data = applied.get_json()["data"]
    assert applied_data["dataset"]["rows"] == 30
    assert applied_data["history"][0]["action"] == "drop"

    reset = client.post(f"/tabular_ml/api/v1/sessions/{session_id}/reset")
    assert reset.get_json()["data"]["dataset"]["rows"] == 31
    assert reset.get_json()["data"]["history"] == []


def test_preparation_route_validates_json_and_target(client) -> None:
    session_id = _session(client)
    no_json = client.post(f"/tabular_ml/api/v1/sessions/{session_id}/preparation/plan")
    no_target = client.post(
        f"/tabular_ml/api/v1/sessions/{session_id}/preparation/plan",
        json={"config": {}},
    )
    assert no_json.get_json()["error"]["code"] == "request.invalid_json"
    assert no_target.get_json()["error"]["code"] == "target.missing"
