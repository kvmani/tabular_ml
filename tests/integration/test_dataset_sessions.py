from __future__ import annotations

import io


def _upload(client, content: bytes, filename: str = "people.csv"):
    return client.post(
        "/tabular_ml/api/v1/datasets/upload",
        data={"file": (io.BytesIO(content), filename)},
        content_type="multipart/form-data",
    )


def test_upload_profile_fetch_and_delete_flow(client) -> None:
    rows = ["person_id,age,team,outcome"]
    rows.extend(
        f"P-{index},{20 + index % 10},{'a' if index % 2 else 'b'},{index % 2}"
        for index in range(30)
    )
    response = _upload(client, ("\n".join(rows) + "\n").encode())
    payload = response.get_json()["data"]

    assert response.status_code == 201
    assert payload["dataset"]["rows"] == 30
    assert payload["profile"]["target_candidates"][0]["column"] == "outcome"
    session_id = payload["session_id"]

    fetched = client.get(f"/tabular_ml/api/v1/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.get_json()["data"]["preview"]["shown"] == 30

    deleted = client.delete(f"/tabular_ml/api/v1/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/tabular_ml/api/v1/sessions/{session_id}").status_code == 404


def test_upload_errors_are_actionable(client) -> None:
    missing = client.post("/tabular_ml/api/v1/datasets/upload")
    unsupported = _upload(client, b"a\n1\n", "demo.tsv")
    unknown = client.post("/tabular_ml/api/v1/datasets/not-real/load")

    assert missing.status_code == 400
    assert missing.get_json()["error"]["code"] == "upload.missing"
    assert unsupported.get_json()["error"]["details"]["accepted_extensions"] == [".csv", ".xlsx"]
    assert unknown.status_code == 404
    assert unknown.get_json()["error"]["code"] == "dataset.not_found"
